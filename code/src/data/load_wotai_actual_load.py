"""Load Wotai actual-load data for strict day-ahead EnergyCA experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/wotai_source/Load Prediction/Preprocessed_Load_Prediction.csv")
TRAIN_START = pd.Timestamp("2025-01-01 00:00:00")
VAL_START = pd.Timestamp("2025-06-04 00:00:00")
TEST_START = pd.Timestamp("2025-07-04 00:00:00")
TEST_END = pd.Timestamp("2025-08-03 23:45:00")
WARMUP_START = pd.Timestamp("2024-12-01 00:00:00")


WEATHER_COLUMNS = [
    "weather_code",
    "temp",
    "pressure",
    "wind_dir",
    "wind_spd",
    "rh",
    "dewpt",
    "clouds",
    "dhi",
    "dni",
    "ghi",
    "solar_rad",
    "elev_angle",
    "azimuth",
    "uv",
]


def _normalise_with_train(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    source = values[train_mask]
    if source.size == 0:
        source = values
    vmin = float(np.nanmin(source))
    vmax = float(np.nanmax(source))
    denom = max(vmax - vmin, 1e-8)
    return ((values - vmin) / denom).astype(np.float32), vmin, vmax


def load_wotai_actual_load(raw_path: Path = RAW_PATH) -> dict:
    """Return a one-node Wotai actual-load dataset with fixed proxy-eval splits.

    The split mirrors the Jiangsu-horizontal proxy load evaluation:
    train starts in 2025, validation is the month before the test window, and
    test targets are 2025-07-04 through 2025-08-03. Validation/test splits keep
    pre-target warmup rows so EnergyCA windows can use settled historical load
    before the target start without using future labels.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Wotai preprocessed load file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if "time" not in df.columns or "actual_load" not in df.columns:
        raise ValueError("Expected columns `time` and `actual_load` in Wotai preprocessed file.")
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.loc[df["time"].notna()].sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last").set_index("time")
    df = df.loc[(df.index >= WARMUP_START) & (df.index <= TEST_END)]

    target = pd.to_numeric(df["actual_load"], errors="coerce").ffill().bfill().fillna(0.0)
    train_norm_mask = (target.index >= TRAIN_START) & (target.index < VAL_START)
    occ_norm, vmin, vmax = _normalise_with_train(target.to_numpy(dtype=np.float32), np.asarray(train_norm_mask))

    weather_cols = [col for col in WEATHER_COLUMNS if col in df.columns]
    weather = df[weather_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill() if weather_cols else pd.DataFrame(index=df.index)

    ts = pd.DatetimeIndex(df.index)
    train_idx = np.flatnonzero((ts >= TRAIN_START) & (ts < VAL_START))
    val_idx = np.flatnonzero((ts >= TRAIN_START) & (ts < TEST_START))
    test_idx = np.flatnonzero((ts >= TRAIN_START) & (ts <= TEST_END))

    node_meta = pd.DataFrame(
        [
            {
                "node_id": "actual_load",
                "site_type": "load",
                "source_column": "actual_load",
                "source_site": "wotai",
                "area": "wotai_proxy",
            }
        ]
    ).set_index("node_id")

    return {
        "occupancy": occ_norm.reshape(-1, 1),
        "occupancy_raw": target.to_numpy(dtype=np.float32).reshape(-1, 1),
        "timestamps": ts,
        "node_ids": ["actual_load"],
        "node_meta": node_meta,
        "adj": np.eye(1, dtype=np.float32),
        "weather": weather,
        "price": pd.DataFrame(index=ts),
        "norm_min": vmin,
        "norm_max": vmax,
        "normalization_source": "train_2025_before_validation",
        "split_indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "split_target_start_times": {
            "val": VAL_START,
            "test": TEST_START,
        },
        "protocol": {
            "train_start": TRAIN_START.isoformat(),
            "val_start": VAL_START.isoformat(),
            "test_start": TEST_START.isoformat(),
            "test_end": TEST_END.isoformat(),
            "target": "actual_load",
        },
    }


if __name__ == "__main__":
    data = load_wotai_actual_load()
    print(f"occupancy shape: {data['occupancy'].shape}")
    print(f"raw range: {data['norm_min']:.3f}..{data['norm_max']:.3f}")
    for split, idx in data["split_indices"].items():
        ts = data["timestamps"][idx]
        print(f"{split}: {len(idx)} rows, {ts[0]}..{ts[-1]}")
