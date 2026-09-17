"""Forecast-origin-safe adapter for official DKASC PV power CSV exports.

The DKA Solar Centre publishes one 5-minute CSV per physical PV array. The
adapter preserves the original exports, aggregates to the paper's hourly PV
protocol, and distinguishes historical measurements from future information.
The experiment builder must therefore use ``future_exog_policy='zero'``: DKASC
contains realised weather observations, not weather forecasts.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/dkasc")
PROCESSED_PATH = Path("data/processed/dkasc_pv.pkl")
TOTAL_PV_PROCESSED_PATH = Path("data/processed/dkasc_total_pv.pkl")

# Copied from the official DKASC Alice Springs download selector on 2026-07-27.
SITE_CATALOG = {
    "100": {
        "node_id": "dkasc_16A",
        "capacity_kw": 2.0,
        "asset": "BP Solar poly-Si fixed north",
        "url": "https://solarcentre.spinifexvalley.com.au/export/100-Site_DKA-M1_A-Phase.csv",
    },
    "103": {
        "node_id": "dkasc_16B",
        "capacity_kw": 2.0,
        "asset": "BP Solar poly-Si fixed flat",
        "url": "https://solarcentre.spinifexvalley.com.au/export/103-Site_DKA-M1_B-Phase.csv",
    },
    "105": {
        "node_id": "dkasc_16C",
        "capacity_kw": 2.0,
        "asset": "BP Solar poly-Si fixed east",
        "url": "https://solarcentre.spinifexvalley.com.au/export/105-Site_DKA-M1_C-Phase.csv",
    },
    "86": {
        "node_id": "dkasc_4",
        "capacity_kw": 2.2,
        "asset": "Kyocera poly-Si dual-axis",
        "url": "https://solarcentre.spinifexvalley.com.au/export/86-Site_DKA-M2_B-Phase.csv",
    },
    "241": {
        "node_id": "dkasc_total_pv",
        "capacity_kw": None,
        "asset": "DKA Solar Centre aggregate photovoltaic generation",
        "url": "https://solarcentre.spinifexvalley.com.au/export/241-Site_DKA_Totals-PV.csv",
    },
}

WEATHER_COLUMNS = (
    "Wind_Speed",
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Wind_Direction",
    "Weather_Daily_Rainfall",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
)


def _raw_files(raw_dir: Path, source_ids: tuple[str, ...]) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for source_id in source_ids:
        if source_id not in SITE_CATALOG:
            raise KeyError(f"Unknown DKASC source ID: {source_id}")
        matches = sorted(raw_dir.glob(f"{source_id}-Site_*.csv"))
        if not matches:
            continue
        path = matches[0]
        # aria2 retains this control file until checksum validation completes.
        if Path(f"{path}.aria2").exists():
            continue
        files.append((source_id, path))
    if not files:
        raise FileNotFoundError(
            f"No completed official DKASC CSV export found under {raw_dir}. "
            "Download one or more SITE_CATALOG URLs with checksum validation first."
        )
    return files


def _file_fingerprint(files: list[tuple[str, Path]]) -> list[dict[str, object]]:
    return [
        {
            "source_id": source_id,
            "name": path.name,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for source_id, path in files
    ]


def _read_hourly(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    available_weather = [column for column in WEATHER_COLUMNS if column in header]
    needed = ["timestamp", "Active_Power", *available_weather]
    # A range-validated official export prefix can end within a quoted CSV
    # record.  The Python parser drops only that incomplete final record while
    # retaining every preceding official observation.
    frame = pd.read_csv(path, usecols=needed, engine="python", on_bad_lines="skip")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    for column in needed[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # The 16A array is rated at 2.0 kW and its official Active_Power peak is
    # about 1.74, so the exported active-power values are interpreted as kW.
    # Negative meter noise is not a valid physical generation target.
    frame["Active_Power"] = frame["Active_Power"].clip(lower=0.0)
    return frame.resample("1h").mean()


def load_dkasc_pv(
    raw_dir: Path = RAW_DIR,
    processed_path: Path = PROCESSED_PATH,
    force_reprocess: bool = False,
    source_ids: tuple[str, ...] = ("100",),
) -> dict:
    """Load selected DKASC exports as an hourly chronological PV benchmark."""
    raw_dir = Path(raw_dir)
    processed_path = Path(processed_path)
    files = _raw_files(raw_dir, source_ids)
    fingerprint = _file_fingerprint(files)
    if processed_path.exists() and not force_reprocess:
        with open(processed_path, "rb") as handle:
            cached = pickle.load(handle)
        if cached.get("source_fingerprint") == fingerprint:
            return cached

    targets: dict[str, pd.Series] = {}
    weather: pd.DataFrame | None = None
    metadata: list[dict[str, object]] = []
    for source_id, path in files:
        hourly = _read_hourly(path)
        item = SITE_CATALOG[source_id]
        node_id = str(item["node_id"])
        targets[node_id] = hourly["Active_Power"]
        if weather is None:
            weather = hourly.drop(columns=["Active_Power"])
        metadata.append({"source_id": source_id, "source_file": str(path), **item})

    target_frame = pd.DataFrame(targets).sort_index()
    weather = (weather if weather is not None else pd.DataFrame(index=target_frame.index)).reindex(target_frame.index)
    # Use the intersection of measured target timestamps across sites. Missing
    # target values are never imputed because they would contaminate evaluation.
    target_frame = target_frame.loc[target_frame.notna().all(axis=1)]
    weather = weather.loc[target_frame.index].ffill().bfill().fillna(0.0)
    if len(target_frame) < 24 * 21:
        raise ValueError("DKASC data are too short for the 168h/24h chronological protocol")

    raw_values = target_frame.to_numpy(dtype=np.float32)
    train_end = max(1, int(len(raw_values) * 0.7))
    train_values = raw_values[:train_end]
    norm_min = float(np.nanmin(train_values))
    norm_max = float(np.nanmax(train_values))
    occupancy = (raw_values - norm_min) / (norm_max - norm_min + 1e-8)
    node_ids = list(target_frame.columns)
    node_meta = pd.DataFrame(metadata).set_index("node_id").reindex(node_ids)

    result = {
        "occupancy": occupancy.astype(np.float32),
        "occupancy_raw": raw_values,
        "timestamps": pd.DatetimeIndex(target_frame.index),
        "node_ids": node_ids,
        "node_meta": node_meta,
        "adj": np.eye(len(node_ids), dtype=np.float32),
        "weather": weather,
        "price": pd.DataFrame(index=target_frame.index),
        "norm_min": norm_min,
        "norm_max": norm_max,
        "normalization_source": "chronological_train_70pct",
        "source_fingerprint": fingerprint,
        "dataset_citation": "DKA Solar Centre, Alice Springs official data download",
        "active_power_unit": "kW (provider Active_Power export; no value rescaling applied)",
        "provider_terms": "https://dkasolarcentre.com.au/download/terms-conditions",
        "future_weather_policy": "measured weather is historical-only; future exogenous values must be zeroed",
    }
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_path, "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_dkasc_total_pv(
    raw_dir: Path = RAW_DIR,
    processed_path: Path = TOTAL_PV_PROCESSED_PATH,
    force_reprocess: bool = False,
) -> dict:
    """Load the official DKA aggregate PV export without mixing array-level files."""
    return load_dkasc_pv(
        raw_dir=raw_dir,
        processed_path=processed_path,
        force_reprocess=force_reprocess,
        source_ids=("241",),
    )
