"""Adverse-weather API workflow for the CD rater.

Mirrors the BEV client batching / error-isolation methodology
(``bev_client.bev_task_batches_threaded``) with CD-rater-specific
defaults:

* **concurrency = 1** (single-threaded payload execution)
* **1 s pause after each wave** (effectively after every payload)
* Automatic per-event retry on batch failure (500 isolation)
* Probability values normalised by dividing by 100

The public entry point is :func:`run_adverse_weather`.
"""

from __future__ import annotations

import os
import time
import json
import datetime
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib3.util.retry import Retry
import requests
from requests.adapters import HTTPAdapter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CD rater forces single-threaded execution with a 1 s inter-wave pause.
_CD_CONCURRENCY = 1
_CD_BATCH_PAUSE_SECONDS = 1.0
ENDPOINT_MAX_COMBINATIONS = {
    "daily": 25,
    "expanding": 10,
}


# Default production base URL (configurable via env or parameter).
_DEFAULT_BASE_URL = (
    "https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth"
)

# Peril sets by endpoint, using production enum names.
DAILY_PERILS: List[str] = ["Rain", "MaxWindSpeed", "MaxWindGust", "Lightning"]
EXPANDING_PERILS: List[str] = ["CumulativeRain"]


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def lookup_locations(
    mapping_df: pd.DataFrame,
    location_ids: Optional[List[int]] = None,
    countries: Optional[List[str]] = None,
    areas: Optional[List[str]] = None,
    cities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter the location mapping by optional criteria.

    All filters that are *not None* are applied (AND logic).  If every
    filter is *None* the full table is returned.
    """
    df = mapping_df.copy()
    if location_ids is not None:
        df = df[df["location_id"].isin(location_ids)]
    if countries is not None:
        df = df[df["country"].isin(countries)]
    if areas is not None:
        df = df[df["area"].isin(areas)]
    if cities is not None:
        df = df[df["city"].isin(cities)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------

def build_events_daily(
    locations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build a list of daily-endpoint event dicts from a locations frame.

    Returns
    -------
    (events, labels)
        *events* – list of dicts ready for the API payload.
        *labels* – parallel list of human-readable location descriptions.
    """
    events: List[Dict[str, Any]] = []
    labels: List[str] = []

    for idx, row in locations_df.iterrows():
        events.append({
            "index": int(idx),
            "location": "",
            "start_date": start_date,
            "end_date": end_date,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        })
        label_parts = [
            str(row.get("city", "")),
            str(row.get("area", "")),
            str(row.get("country", "")),
        ]
        label = ", ".join(p for p in label_parts if p)
        labels.append(f"{label} ({row['latitude']}, {row['longitude']})")

    return events, labels


def build_events_expanding(
    locations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    start_hour: int = 0,
    end_hour: int = 23,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build a list of expanding-endpoint event dicts from a locations frame.

    Same as :func:`build_events_daily` but adds ``start_hour``,
    ``end_hour`` and ``tag`` fields required by the expanding endpoint.
    """
    events: List[Dict[str, Any]] = []
    labels: List[str] = []

    for idx, row in locations_df.iterrows():
        events.append({
            "index": int(idx),
            "tag": f"tag-{idx}",
            "location": "",
            "start_date": start_date,
            "end_date": end_date,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        })
        label_parts = [
            str(row.get("city", "")),
            str(row.get("area", "")),
            str(row.get("country", "")),
        ]
        label = ", ".join(p for p in label_parts if p)
        labels.append(f"{label} ({row['latitude']}, {row['longitude']})")

    return events, labels


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------

def normalise_daily_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten and normalise daily API responses into the CD rater schema.

    The production API returns probability values on a 0-100 scale
    (e.g. ``82``).  This function divides by 100 so the value stored in
    the CD rater schema is ``0.82``.

    Returns a list of dicts matching the ``adverse_weather_daily`` schema::

        {"peril": str, "index": int, "threshold": str, "value": float}
    """
    rows: List[Dict[str, Any]] = []
    for r in raw_results:
        idx = r.get("index")
        peril = r.get("peril")

        # Prod format: parallel threshold / probability arrays.
        thresholds = r.get("threshold") or []
        probabilities = r.get("probability") or []

        for t, p in zip(thresholds, probabilities):
            try:
                value = float(p) / 100.0
            except (TypeError, ValueError):
                value = None
            rows.append({
                "peril": peril,
                "index": idx,
                "threshold": str(t),
                "value": value,
            })

        # Legacy format: model dict mapping threshold -> value.
        if "model" in r and isinstance(r.get("model"), dict):
            for thresh, val in r["model"].items():
                try:
                    value = float(val) / 100.0
                except (TypeError, ValueError):
                    value = None
                rows.append({
                    "peril": peril,
                    "index": idx,
                    "threshold": str(thresh),
                    "value": value,
                })

    return rows


def normalise_expanding_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten and normalise expanding API responses into the CD rater schema.

    Returns a list of dicts matching the ``adverse_weather_expanding`` schema::

        {"index": int, "window_index": int, "peril": str,
         "threshold": float, "value": float}

    Probability values are divided by 100 (same as daily).
    """
    rows: List[Dict[str, Any]] = []
    for r in raw_results:
        try:
            value = float(r.get("value", 0)) / 100.0
        except (TypeError, ValueError):
            value = None
        try:
            threshold = float(r.get("threshold", 0))
        except (TypeError, ValueError):
            threshold = None
        rows.append({
            "index": r.get("index"),
            "window_index": r.get("window_index"),
            "peril": r.get("peril"),
            "threshold": threshold,
            "value": value,
        })
    return rows


# ---------------------------------------------------------------------------
# Failed-event warning helpers
# ---------------------------------------------------------------------------

def build_failed_warnings(
    failed_events: List[Dict[str, Any]],
    labels: List[str],
    endpoint: str,
) -> List[Dict[str, str]]:
    """Build warning dicts for each failed event.

    Returns a list of ``{"label": ..., "message": ...}`` dicts suitable
    for appending to ``hxd.outputs.warnings``.
    """
    warnings: List[Dict[str, str]] = []
    for fe in failed_events:
        ev = fe["event"]
        idx = ev.get("index")
        lat = ev.get("latitude")
        lon = ev.get("longitude")
        label = labels[idx] if isinstance(idx, int) and idx < len(labels) else "N/A"
        warnings.append({
            "label": f"Adverse weather /{endpoint} failed",
            "message": (
                f"index={idx} lat={lat} lon={lon} label={label} — {fe['error']}"
            ),
        })
    return warnings


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------
def _json_converter(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def batch_payload(
    perils: List[Any],
    events: List[Dict[str, Any]],
    window_days: Optional[int] = None,
    max_combinations: int = 4,
):
    """Batch perils and events into smaller payloads to respect API limits."""
    if not perils:
        return []
    # max_events_per_batch = max(1, max_combinations // len(perils))
    max_events_per_batch = max(1, max_combinations // len(perils))

    if window_days is not None:
        return [
            {
                "perils": perils,
                "events": events[i : i + max_events_per_batch],
                "window_days": window_days,
            }
            for i in range(0, len(events), max_events_per_batch)
        ]
    else:
        return [
            {"perils": perils, "events": events[i : i + max_events_per_batch]}
            for i in range(0, len(events), max_events_per_batch)
        ]


def _post_payload(
    session: requests.Session,
    url: str,
    headers: dict,
    payload: dict,
    timeout: float,
):
    """POST a single payload and return parsed JSON; raises for HTTP errors."""
    json_payload = json.dumps(payload, default=_json_converter)
    resp = session.post(url, headers=headers, data=json_payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()  # API returns a list per payload


def bev_task_batches_threaded(
    api_key: str,
    perils: List[Any],
    endpoint: str,
    event_set: List[Dict[str, Any]],
    window_days: Optional[int] = None,
    concurrency: int = 30,
    request_timeout: float = 300.0,
    max_retries: int = 3,
    verify_ssl: bool = False,
    ca_bundle: Optional[str] = None,
    base_url: Optional[str] = None,
    endpoint_trailing: bool = False,
    max_combinations: Optional[int] = None,
    batch_pause_seconds: float = 0,
):
    """Call BEV weather API in parallel batches and return merged results.

    Payloads are split so that no single request exceeds the provider's
    event-peril combination limit (25 for /daily, 10 for /expanding by
    default).  Payloads are sent in waves of ``concurrency`` requests;
    between waves the caller can inject a pause via ``batch_pause_seconds``
    to respect rate limits.

    Args:
        api_key: API key for authentication.
        perils: Perils configuration; see endpoint schema.
        endpoint: 'daily' or 'expanding'.
        event_set: List of event dicts with keys like 'index', 'location',
            'latitude', 'longitude', 'start_date', 'end_date'.
        window_days: For 'expanding' endpoint only.
        concurrency: Max concurrent requests per wave.
        request_timeout: Per-request timeout in seconds.
        max_retries: Retry count for transient errors.
        verify_ssl: Whether to verify SSL certificates.
        ca_bundle: Optional path to a CA bundle file (overrides verify_ssl).
        base_url: Optional base URL for the API. Defaults to staging URL.
        endpoint_trailing: If True, append trailing slash to endpoint URL.
        max_combinations: Max event-peril combinations per request.
            Defaults to 25 for /daily and 10 for /expanding.
        batch_pause_seconds: Seconds to sleep between concurrency waves.
            Set to 0 (default) to disable pausing.

    Returns:
        tuple: ``(results, failed_events)`` where *results* is the combined
        list of API responses and *failed_events* is a list of dicts
        ``{"event": <event_dict>, "error": <str>}`` for locations that
        returned 500 errors even after individual retries.
    """

    # Construct URL; the old url used to have a trailing "/" 
    # whereas the new one doesn't this functionality
    # is legacy from that and can be deleted after testing.
    if endpoint_trailing:
        url = base_url.rstrip("/") + "/" + endpoint + "/"
    else:
        url = base_url.rstrip("/") + "/" + endpoint

    json_data = {"perils": perils, "events": event_set}

    payloads = batch_payload(
        json_data["perils"],
        json_data["events"],
        window_days=window_days,
        max_combinations=max_combinations,
    )


    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "accept-encoding": "gzip",
    }

    # This part stops the task from crashing if an error is returned.
    # Creates a Retry policy that will retry POST requests up to max_retries times, 
    # backing off exponentially (starting at 0.5 s) when the response status is 429/500/502/503/504.
    retries = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    # Builds an HTTPAdapter (for requests sessions) configured with that retry policy 
    # and with connection pooling sized by concurrency for both total connections 
    # and per‑pool max.
    adapter = HTTPAdapter(
        pool_connections=max(1, concurrency),
        pool_maxsize=max(1, concurrency),
        max_retries=retries,
    )

    session = requests.Session()
    session.headers.update(headers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Support passing a CA bundle path (string) or a boolean verify flag
    if ca_bundle is not None:
        session.verify = ca_bundle
    else:
        session.verify = verify_ssl

    # Now the request is configured we start with the api calls
    results = []
    failed_events = []

    # Send payloads in waves of `concurrency`, pausing between waves.
    wave_size = max(1, concurrency)
    for wave_start in range(0, len(payloads), wave_size):
        wave = payloads[wave_start : wave_start + wave_size]

        with ThreadPoolExecutor(max_workers=wave_size) as executor:
            future_to_payload = {
                executor.submit(
                    _post_payload, session, url, headers, pl, request_timeout
                ): pl
                for pl in wave
            }
            for fut in as_completed(future_to_payload):
                try:
                    r = fut.result()
                    results.extend(r)
                except Exception as exc:
                    # Batch failed – retry each event individually to isolate
                    # the problematic location(s).
                    failed_payload = future_to_payload[fut]
                    batch_events = failed_payload["events"]
                    for event in batch_events:
                        single_payload = {
                            "perils": failed_payload["perils"],
                            "events": [event],
                        }
                        if "window_days" in failed_payload:
                            single_payload["window_days"] = failed_payload[
                                "window_days"
                            ]
                        try:
                            r = _post_payload(
                                session, url, headers, single_payload,
                                request_timeout,
                            )
                            results.extend(r)
                        except Exception as single_exc:
                            failed_events.append({
                                "event": event,
                                "error": str(single_exc),
                            })

        # Pause between waves (skip after the last wave)
        if batch_pause_seconds > 0 and wave_start + wave_size < len(payloads):
            time.sleep(batch_pause_seconds)


    return results, failed_events

def run_adverse_weather(
    locations_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    perils_daily: Optional[List[str]] = None,
    perils_expanding: Optional[List[str]] = None,
    window_days: int = 0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verify_ssl: bool = False,
    ca_bundle: Optional[str] = None,
    max_retries: int = 3,
    request_timeout: float = 300.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Run the adverse-weather API workflow for the CD rater.

    Returns
    -------
    (daily_rows, expanding_rows, failed_events, warnings)
        *daily_rows* matches the ``adverse_weather_daily`` schema.
        *expanding_rows* matches the ``adverse_weather_expanding`` schema.
        *failed_events* is a list of
        ``{"event": <dict>, "error": <str>}`` dicts.
        *warnings* is a list of ``{"label": ..., "message": ...}`` dicts
        ready for ``hxd.outputs.warnings``.
    """
    # Resolve defaults
    if api_key is None:
        api_key = os.environ.get("BEV_API_KEY_PROD") or os.environ.get("BEV_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key provided and BEV_API_KEY_PROD / BEV_API_KEY not set."
        )

    if base_url is None:
        base_url = os.environ.get("BEV_BASE_URL", _DEFAULT_BASE_URL)

    today = date.today().isoformat()
    start_date = start_date or today
    end_date = end_date or today

    if perils_daily is None:
        perils_daily = list(DAILY_PERILS)
    if perils_expanding is None:
        perils_expanding = list(EXPANDING_PERILS)

    all_failed: List[Dict[str, Any]] = []
    all_warnings: List[Dict[str, str]] = []
    daily_rows: List[Dict[str, Any]] = []
    expanding_rows: List[Dict[str, Any]] = []

    # --- Daily endpoint ------------------------------------------------
    if perils_daily:
        events_daily, labels_daily = build_events_daily(
            locations_df, start_date, end_date,
        )

        raw_daily, failed_daily = bev_task_batches_threaded(
            api_key=api_key,
            perils=perils_daily,
            endpoint="daily",
            event_set=events_daily,
            concurrency=_CD_CONCURRENCY,
            request_timeout=request_timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
            base_url=base_url,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["daily"],
            batch_pause_seconds=_CD_BATCH_PAUSE_SECONDS,
        )

        daily_rows = normalise_daily_results(raw_daily)
        all_failed.extend(failed_daily)
        all_warnings.extend(build_failed_warnings(failed_daily, labels_daily, "daily"))

    # --- Expanding endpoint --------------------------------------------
    if perils_expanding:
        events_expanding, labels_expanding = build_events_expanding(
            locations_df, start_date, end_date,
        )

        raw_expanding, failed_expanding = bev_task_batches_threaded(
            api_key=api_key,
            perils=perils_expanding,
            endpoint="expanding",
            event_set=events_expanding,
            window_days=window_days,
            concurrency=_CD_CONCURRENCY,
            request_timeout=request_timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
            base_url=base_url,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["expanding"],
            batch_pause_seconds=_CD_BATCH_PAUSE_SECONDS,
        )

        expanding_rows = normalise_expanding_results(raw_expanding)
        all_failed.extend(failed_expanding)
        all_warnings.extend(build_failed_warnings(failed_expanding, labels_expanding, "expanding"))

    return daily_rows, expanding_rows, all_failed, all_warnings

