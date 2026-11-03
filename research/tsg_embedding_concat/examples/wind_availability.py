"""Generate two synthetic wind observation windows at the same forecast origin."""

from datetime import datetime, timedelta, timezone
import json


def wind_window(origin, delay_latest):
    cadence = timedelta(minutes=10)
    history = []
    for index in range(144):
        observed_at = origin - (143 - index) * cadence
        received_at = origin + cadence if delay_latest and index >= 142 else observed_at
        history.append({
            "observed_at": observed_at.isoformat(),
            "received_at": received_at.isoformat(),
            "power_kw": 400 + index if received_at <= origin else None,
        })

    mask = [row["power_kw"] is not None for row in history]
    latest = next(row for row in reversed(history) if row["power_kw"] is not None)
    age = (origin - datetime.fromisoformat(latest["observed_at"])) // timedelta(minutes=1)
    count = sum(mask)
    return {
        "task": "wind",
        "asset": "synthetic_turbine",
        "origin": origin.isoformat(),
        "cadence_minutes": 10,
        "history": history,
        "history_mask": mask,
        "n24": count,
        "latest_operation_age_minutes": age,
        "availability_text": f"latest_operation_age_minutes={age}; n24={count}",
    }


if __name__ == "__main__":
    origin = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)
    print(json.dumps({
        "on_time": wind_window(origin, delay_latest=False),
        "delayed": wind_window(origin, delay_latest=True),
    }, indent=2))
