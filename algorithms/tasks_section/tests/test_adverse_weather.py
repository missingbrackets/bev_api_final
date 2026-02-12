import os
import json
from typing import Optional

import pandas as pd
import requests

from algorithms.tasks_section import adverse_weather as module


STAGE_URL = 'https://nonprodstage-weather-api-wrapper.birdseyeviewtechnologies.com/model/daily'


def check_connection(url: str) -> None:
    try:
        resp = requests.get(url, timeout=10)
        print(f"Connectivity check for {url}:", resp.status_code, resp.text[:200])
        resp.raise_for_status()
    except Exception as exc:
        print(f"Connectivity check for {url} failed:", exc)


def run_build_events_daily_example():
    locations_df = pd.DataFrame(
        {
            "latitude": [51.5074, 40.7128],
            "longitude": [-0.1278, -74.0060],
            "city": ["London", "New York"],
            "area": ["England", "New York"],
            "country": ["United Kingdom", "United States"],
        }
    )
    events, labels = module.build_events_daily(locations_df, "2024-01-01", "2024-01-02")
    print("Daily events:", json.dumps(events, indent=2))
    print("Daily labels:", labels)


def run_normalise_daily_results_example():
    raw = [
        {
            "index": 3,
            "peril": "Rain",
            "threshold": [10],
            "probability": [25],
        },
        {
            "index": 4,
            "peril": "Rain",
            "model": {"15": "40"},
        },
    ]
    rows = module.normalise_daily_results(raw)
    print("Normalised daily results:", json.dumps(rows, indent=2))


def run_full_adverse_weather(api_key: str, base_url: Optional[str] = None):
    locations_df = pd.DataFrame(
        {
            "latitude": [35.0],
            "longitude": [-90.0],
            "city": ["Springfield"],
            "area": ["State"],
            "country": ["Country"],
        }
    )

    def logging_post_payload(session, url, headers, payload, timeout):
        print("POST", url)
        safe_headers = {k: ("***" if k.lower() == "x-api-key" else v) for k, v in headers.items()}
        print("Headers:", safe_headers)
        print("Payload:", json.dumps(payload, indent=2))
        json_payload = json.dumps(payload)
        resp = session.post(url, headers=headers, data=json_payload, timeout=timeout)
        print("Status code:", resp.status_code)
        print("Response body:", resp.text[:2000])
        resp.raise_for_status()
        return resp.json()

    original_post_payload = module._post_payload
    module._post_payload = logging_post_payload
    try:
        daily_rows, expanding_rows, failed_events, warnings = module.run_adverse_weather(
            locations_df=locations_df,
            start_date="2024-01-01",
            end_date="2024-01-02",
            window_days=3,
            api_key=api_key,
            base_url=base_url,
        )
    finally:
        module._post_payload = original_post_payload

    #print("Daily rows:", json.dumps(daily_rows, indent=2))
    #print("Expanding rows:", json.dumps(expanding_rows, indent=2))
    print("Failed events:", json.dumps(failed_events, indent=2))
    print("Warnings:", json.dumps(warnings, indent=2))


def main():
    base_url = os.environ.get("BEV_BASE_URL", module._DEFAULT_BASE_URL)
    #check_connection(base_url)
    #check_connection(STAGE_URL)
    run_build_events_daily_example()
    run_normalise_daily_results_example()
    api_key = os.environ.get("BEV_API_KEY_PROD") or os.environ.get("BEV_API_KEY")
    if not api_key:
        raise SystemExit("Set BEV_API_KEY_PROD or BEV_API_KEY before running.")
    run_full_adverse_weather(api_key=api_key, base_url=base_url)
    print("Adverse weather execution completed.")


if __name__ == "__main__":
    main()
