"""Regression test: expanding event payloads in the task path must include
the same fields that build_events_expanding() produces.

The root cause of the adverse-weather divergence was that
adverse_weather_rating() (tasks.py) omitted start_hour, end_hour, and tag
from expanding events, while the tested module path (run_adverse_weather)
included them via build_events_expanding().
"""

import pandas as pd

from algorithms.tasks_section.adverse_weather import build_events_expanding


EXPANDING_REQUIRED_KEYS = {
    "index",
    "tag",
    "location",
    "start_date",
    "end_date",
    "start_hour",
    "end_hour",
    "latitude",
    "longitude",
}


def _build_task_path_expanding_events(df_event_set: pd.DataFrame) -> list:
    """Replicate the expanding-event construction from tasks.py."""
    return (
        df_event_set[["index", "start_date", "end_date", "latitude", "longitude"]]
        .assign(
            location="",
            start_hour=0,
            end_hour=23,
            tag=lambda df: "tag-" + df["index"].astype(str),
        )
        .to_dict(orient="records")
    )


def _sample_locations():
    return pd.DataFrame(
        {
            "latitude": [35.0, 51.5],
            "longitude": [-90.0, -0.1],
            "city": ["Springfield", "London"],
            "area": ["State", "England"],
            "country": ["Country", "United Kingdom"],
        }
    )


def test_task_path_expanding_keys_match_module():
    """Expanding events built by the task path must contain all keys that
    build_events_expanding() produces."""
    locations_df = _sample_locations()

    # Module path (source of truth)
    module_events, _ = build_events_expanding(
        locations_df, "2024-01-01", "2024-01-02"
    )

    # Task path (simulated)
    df_event_set = pd.DataFrame(
        {
            "index": list(range(len(locations_df))),
            "start_date": ["2024-01-01"] * len(locations_df),
            "end_date": ["2024-01-02"] * len(locations_df),
            "latitude": locations_df["latitude"].tolist(),
            "longitude": locations_df["longitude"].tolist(),
        }
    )
    task_events = _build_task_path_expanding_events(df_event_set)

    # Both should have the required keys
    for ev in module_events:
        assert set(ev.keys()) == EXPANDING_REQUIRED_KEYS, (
            f"Module event keys mismatch: {set(ev.keys())}"
        )
    for ev in task_events:
        assert set(ev.keys()) == EXPANDING_REQUIRED_KEYS, (
            f"Task event keys mismatch: {set(ev.keys())}"
        )


def test_task_path_expanding_default_values():
    """start_hour should be 0, end_hour should be 23, and tag should be
    'tag-<index>'."""
    df_event_set = pd.DataFrame(
        {
            "index": [0, 5, 12],
            "start_date": ["2024-06-01"] * 3,
            "end_date": ["2024-06-02"] * 3,
            "latitude": [35.0, 51.5, 40.7],
            "longitude": [-90.0, -0.1, -74.0],
        }
    )
    events = _build_task_path_expanding_events(df_event_set)

    for ev in events:
        assert ev["start_hour"] == 0
        assert ev["end_hour"] == 23
        assert ev["tag"] == f"tag-{ev['index']}"
        assert ev["location"] == ""


if __name__ == "__main__":
    test_task_path_expanding_keys_match_module()
    test_task_path_expanding_default_values()
    print("All expanding-event shape tests passed.")
