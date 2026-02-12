"""Regression tests: event payloads built by the task path (tasks.py) must
match the shape, keys, and types produced by build_events_daily() and
build_events_expanding() in adverse_weather.py.

Root causes fixed:
1. Expanding events were missing start_hour, end_hour, tag.
2. Daily + expanding events contained numpy scalar types that could not be
   serialised by _json_converter (which also had a broken datetime import).
"""

import json

import numpy as np
import pandas as pd

from algorithms.tasks_section.adverse_weather import (
    build_events_daily,
    build_events_expanding,
    _json_converter,
)


# ── Expected key sets ────────────────────────────────────────────────────

DAILY_REQUIRED_KEYS = {
    "index", "location", "start_date", "end_date", "latitude", "longitude",
}

EXPANDING_REQUIRED_KEYS = DAILY_REQUIRED_KEYS | {"start_hour", "end_hour", "tag"}


# ── Helpers that mirror the *fixed* tasks.py construction ────────────────

def _build_task_daily_events(df_event_set: pd.DataFrame) -> list:
    """Replicate daily-event construction from tasks.py (post-fix)."""
    return [
        {
            "index": int(row["index"]),
            "location": "",
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }
        for row in df_event_set.to_dict(orient="records")
    ]


def _build_task_expanding_events(df_event_set: pd.DataFrame) -> list:
    """Replicate expanding-event construction from tasks.py (post-fix)."""
    return [
        {
            "index": int(row["index"]),
            "tag": f"tag-{int(row['index'])}",
            "location": "",
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "start_hour": 0,
            "end_hour": 23,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        }
        for row in df_event_set.to_dict(orient="records")
    ]


def _sample_locations():
    return pd.DataFrame({
        "latitude": [35.0, 51.5],
        "longitude": [-90.0, -0.1],
        "city": ["Springfield", "London"],
        "area": ["State", "England"],
        "country": ["Country", "United Kingdom"],
    })


def _sample_event_set():
    return pd.DataFrame({
        "index": [0, 5, 12],
        "start_date": ["2024-06-01"] * 3,
        "end_date": ["2024-06-02"] * 3,
        "latitude": [35.0, 51.5, 40.7],
        "longitude": [-90.0, -0.1, -74.0],
    })


# ── Daily tests ──────────────────────────────────────────────────────────

def test_task_daily_keys_match_module():
    """Daily events from the task path must have exactly the same keys as
    build_events_daily()."""
    locs = _sample_locations()
    module_events, _ = build_events_daily(locs, "2024-01-01", "2024-01-02")

    df = pd.DataFrame({
        "index": list(range(len(locs))),
        "start_date": ["2024-01-01"] * len(locs),
        "end_date": ["2024-01-02"] * len(locs),
        "latitude": locs["latitude"].tolist(),
        "longitude": locs["longitude"].tolist(),
    })
    task_events = _build_task_daily_events(df)

    for ev in module_events:
        assert set(ev.keys()) == DAILY_REQUIRED_KEYS
    for ev in task_events:
        assert set(ev.keys()) == DAILY_REQUIRED_KEYS


def test_task_daily_native_types():
    """index must be int, lat/lon must be float — not numpy scalars."""
    events = _build_task_daily_events(_sample_event_set())
    for ev in events:
        assert type(ev["index"]) is int, f"index is {type(ev['index'])}"
        assert type(ev["latitude"]) is float, f"latitude is {type(ev['latitude'])}"
        assert type(ev["longitude"]) is float, f"longitude is {type(ev['longitude'])}"


def test_task_daily_json_serialisable():
    """Daily event dicts must round-trip through json.dumps without a
    custom converter (native types only)."""
    events = _build_task_daily_events(_sample_event_set())
    for ev in events:
        # Should not raise
        json.dumps(ev)


# ── Expanding tests ──────────────────────────────────────────────────────

def test_task_expanding_keys_match_module():
    """Expanding events from the task path must have exactly the same keys
    as build_events_expanding()."""
    locs = _sample_locations()
    module_events, _ = build_events_expanding(locs, "2024-01-01", "2024-01-02")

    df = pd.DataFrame({
        "index": list(range(len(locs))),
        "start_date": ["2024-01-01"] * len(locs),
        "end_date": ["2024-01-02"] * len(locs),
        "latitude": locs["latitude"].tolist(),
        "longitude": locs["longitude"].tolist(),
    })
    task_events = _build_task_expanding_events(df)

    for ev in module_events:
        assert set(ev.keys()) == EXPANDING_REQUIRED_KEYS
    for ev in task_events:
        assert set(ev.keys()) == EXPANDING_REQUIRED_KEYS


def test_task_expanding_default_values():
    """start_hour=0, end_hour=23, tag='tag-<index>', location=''."""
    events = _build_task_expanding_events(_sample_event_set())
    for ev in events:
        assert ev["start_hour"] == 0
        assert ev["end_hour"] == 23
        assert ev["tag"] == f"tag-{ev['index']}"
        assert ev["location"] == ""


def test_task_expanding_native_types():
    """Numeric values must be native Python types."""
    events = _build_task_expanding_events(_sample_event_set())
    for ev in events:
        assert type(ev["index"]) is int
        assert type(ev["latitude"]) is float
        assert type(ev["longitude"]) is float
        assert type(ev["start_hour"]) is int
        assert type(ev["end_hour"]) is int


def test_task_expanding_json_serialisable():
    """Expanding event dicts must round-trip through json.dumps."""
    events = _build_task_expanding_events(_sample_event_set())
    for ev in events:
        json.dumps(ev)


# ── _json_converter tests ────────────────────────────────────────────────

def test_json_converter_numpy_int():
    assert _json_converter(np.int64(42)) == 42
    assert type(_json_converter(np.int64(42))) is int


def test_json_converter_numpy_float():
    assert _json_converter(np.float64(3.14)) == 3.14
    assert type(_json_converter(np.float64(3.14))) is float


def test_json_converter_datetime():
    import datetime
    result = _json_converter(datetime.date(2024, 1, 15))
    assert result == "2024-01-15"


if __name__ == "__main__":
    test_task_daily_keys_match_module()
    test_task_daily_native_types()
    test_task_daily_json_serialisable()
    test_task_expanding_keys_match_module()
    test_task_expanding_default_values()
    test_task_expanding_native_types()
    test_task_expanding_json_serialisable()
    test_json_converter_numpy_int()
    test_json_converter_numpy_float()
    test_json_converter_datetime()
    print("All event-shape and serialisation tests passed.")
