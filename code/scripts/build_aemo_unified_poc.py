#!/usr/bin/env python
"""Build an auditable AEMO same-system four-task proof of concept."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = ("NSW1", "QLD1", "VIC1", "SA1", "TAS1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--month", default="2024-01")
    parser.add_argument("--forecast_origin_hour", type=int, default=0)
    parser.add_argument("--origin_start_day", type=int, default=1)
    parser.add_argument("--origin_end_buffer_days", type=int, default=7)
    return parser.parse_args()


def archive(raw_dir: Path, table: str) -> Path:
    matches = sorted(raw_dir.glob(f"PUBLIC_DVD_{table}_*.zip"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {table} archive in {raw_dir}, found {len(matches)}")
    return matches[0]


def read_mms(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as payload:
        member = payload.namelist()[0]
        with payload.open(member) as stream:
            stream.readline()
            header = stream.readline().decode("utf-8-sig").strip().split(",")[4:]
        missing = sorted(set(columns) - set(header))
        if missing:
            raise ValueError(f"{path.name} is missing fields {missing}")
        names = ["record_type", "package", "table", "version", *header]
        with payload.open(member) as stream:
            frame = pd.read_csv(
                stream,
                header=None,
                skiprows=2,
                names=names,
                usecols=["record_type", *columns],
                dtype=str,
                low_memory=False,
            )
    return frame.loc[frame["record_type"].eq("D"), columns].copy()


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def half_hour_mean(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "REGIONID",
                pd.Grouper(key="timestamp", freq="30min", label="right", closed="right"),
            ],
            observed=True,
        )[value]
        .mean()
        .rename(value)
        .reset_index()
    )


def active_unit_map(raw_dir: Path, month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
    details = read_mms(
        archive(raw_dir, "DUDETAILSUMMARY"),
        ["DUID", "START_DATE", "END_DATE", "REGIONID"],
    )
    details["START_DATE"] = pd.to_datetime(details["START_DATE"])
    details["END_DATE"] = pd.to_datetime(details["END_DATE"], errors="coerce")
    details = details.loc[
        details["START_DATE"].le(month_end)
        & (details["END_DATE"].isna() | details["END_DATE"].ge(month_start))
        & details["REGIONID"].isin(REGIONS)
    ]
    details = details.sort_values("START_DATE").drop_duplicates("DUID", keep="last")

    units = read_mms(
        archive(raw_dir, "GENUNITS"),
        ["GENSETID", "REGISTEREDCAPACITY", "CO2E_ENERGY_SOURCE"],
    )
    units = numeric(units, ["REGISTEREDCAPACITY"])
    units = units.loc[units["CO2E_ENERGY_SOURCE"].isin(["Wind", "Solar"])]
    return details.merge(units, left_on="DUID", right_on="GENSETID", how="inner")


def build_actuals(raw_dir: Path, month_start: pd.Timestamp, month_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    dispatch = read_mms(
        archive(raw_dir, "DISPATCHREGIONSUM"),
        ["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "TOTALDEMAND"],
    )
    dispatch = numeric(dispatch, ["INTERVENTION", "TOTALDEMAND"])
    dispatch["timestamp"] = pd.to_datetime(dispatch["SETTLEMENTDATE"])
    dispatch = dispatch.loc[
        dispatch["REGIONID"].isin(REGIONS)
        & dispatch["INTERVENTION"].eq(0)
        & dispatch["timestamp"].gt(month_start)
        & dispatch["timestamp"].le(month_end)
    ]
    demand = half_hour_mean(dispatch, "TOTALDEMAND").rename(columns={"TOTALDEMAND": "operational_demand"})

    unit_map = active_unit_map(raw_dir, month_start, month_end)
    scada = read_mms(
        archive(raw_dir, "DISPATCH_UNIT_SCADA"),
        ["SETTLEMENTDATE", "DUID", "SCADAVALUE"],
    )
    scada = numeric(scada, ["SCADAVALUE"])
    scada["timestamp"] = pd.to_datetime(scada["SETTLEMENTDATE"])
    scada = scada.loc[scada["timestamp"].gt(month_start) & scada["timestamp"].le(month_end)]
    scada = scada.merge(
        unit_map[["DUID", "REGIONID", "CO2E_ENERGY_SOURCE"]],
        on="DUID",
        how="inner",
    )
    generation_5min = (
        scada.groupby(["timestamp", "REGIONID", "CO2E_ENERGY_SOURCE"], observed=True)["SCADAVALUE"]
        .sum()
        .reset_index()
    )
    generation_30min = (
        generation_5min.groupby(
            [
                "REGIONID",
                "CO2E_ENERGY_SOURCE",
                pd.Grouper(key="timestamp", freq="30min", label="right", closed="right"),
            ],
            observed=True,
        )["SCADAVALUE"]
        .mean()
        .unstack("CO2E_ENERGY_SOURCE", fill_value=0.0)
        .rename(columns={"Wind": "wind", "Solar": "utility_pv"})
        .reset_index()
    )

    rooftop = read_mms(
        archive(raw_dir, "ROOFTOP_PV_ACTUAL"),
        ["INTERVAL_DATETIME", "REGIONID", "POWER", "QI", "TYPE"],
    )
    rooftop = numeric(rooftop, ["POWER", "QI"])
    rooftop["timestamp"] = pd.to_datetime(rooftop["INTERVAL_DATETIME"])
    rooftop = rooftop.loc[
        rooftop["REGIONID"].isin(REGIONS)
        & rooftop["timestamp"].gt(month_start)
        & rooftop["timestamp"].le(month_end)
    ]
    rooftop["source_priority"] = rooftop["TYPE"].map({"MEASUREMENT": 0, "SATELLITE": 1}).fillna(2)
    rooftop = (
        rooftop.sort_values(["timestamp", "REGIONID", "source_priority", "QI"], ascending=[True, True, True, False])
        .drop_duplicates(["timestamp", "REGIONID"], keep="first")
        [["timestamp", "REGIONID", "POWER"]]
        .rename(columns={"POWER": "rooftop_pv"})
    )

    actuals = demand.merge(generation_30min, on=["timestamp", "REGIONID"], how="left")
    generation_columns = ["wind", "utility_pv"]
    missing_generation = actuals[generation_columns].isna().any(axis=1)
    actuals = actuals.sort_values(["REGIONID", "timestamp"])
    for column in generation_columns:
        actuals[column] = actuals.groupby("REGIONID", sort=False)[column].transform(
            lambda values: values.interpolate(method="linear", limit=1, limit_area="inside")
        )
    remaining_generation_missing = actuals[generation_columns].isna().any(axis=1)
    if remaining_generation_missing.any():
        first = actuals.loc[remaining_generation_missing, ["timestamp", "REGIONID"]].iloc[0].to_dict()
        raise ValueError(f"AEMO generation SCADA contains a gap longer than one half-hour: {first}")
    actuals["generation_imputed"] = missing_generation.reindex(actuals.index).to_numpy(dtype=bool)
    actuals = actuals.merge(rooftop, on=["timestamp", "REGIONID"], how="left")
    actuals["rooftop_pv"] = actuals["rooftop_pv"].fillna(0.0)
    actuals["load"] = actuals["operational_demand"] + actuals["rooftop_pv"]
    actuals["pv"] = actuals["utility_pv"] + actuals["rooftop_pv"]
    actuals["net_load"] = actuals["load"] - actuals["wind"] - actuals["pv"]
    actuals = actuals.sort_values(["timestamp", "REGIONID"]).reset_index(drop=True)

    capacity = (
        unit_map.groupby(["REGIONID", "CO2E_ENERGY_SOURCE"], observed=True)["REGISTEREDCAPACITY"]
        .sum()
        .unstack("CO2E_ENERGY_SOURCE", fill_value=0.0)
        .rename(columns={"Wind": "wind_capacity", "Solar": "utility_pv_capacity"})
        .reset_index()
    )
    return actuals, capacity


def latest_at_origin(frame: pd.DataFrame, origin: pd.Timestamp, version_column: str) -> pd.DataFrame:
    selected = frame.loc[
        frame[version_column].le(origin)
        & frame["valid_time"].gt(origin)
        & frame["valid_time"].le(origin + pd.Timedelta(days=7))
    ]
    return selected.sort_values(version_column).drop_duplicates(["REGIONID", "valid_time"], keep="last")


def build_issued_forecasts(
    raw_dir: Path,
    actuals: pd.DataFrame,
    month_start: pd.Timestamp,
    month_end: pd.Timestamp,
    forecast_origin_hour: int,
    origin_start_day: int = 1,
    origin_end_buffer_days: int = 7,
) -> pd.DataFrame:
    stpasa = read_mms(
        archive(raw_dir, "STPASA_REGIONSOLUTION"),
        [
            "RUN_DATETIME",
            "INTERVAL_DATETIME",
            "REGIONID",
            "DEMAND50",
            "SS_SOLAR_UIGF",
            "SS_WIND_UIGF",
            "RUNTYPE",
        ],
    )
    stpasa = numeric(stpasa, ["DEMAND50", "SS_SOLAR_UIGF", "SS_WIND_UIGF"])
    stpasa["run_time"] = pd.to_datetime(stpasa["RUN_DATETIME"])
    stpasa["valid_time"] = pd.to_datetime(stpasa["INTERVAL_DATETIME"])
    stpasa = stpasa.loc[
        stpasa["REGIONID"].isin(REGIONS)
        & stpasa["RUNTYPE"].eq("OUTAGE_LRC")
        & stpasa["DEMAND50"].notna()
    ]

    rooftop = read_mms(
        archive(raw_dir, "ROOFTOP_PV_FORECAST"),
        ["VERSION_DATETIME", "INTERVAL_DATETIME", "REGIONID", "POWERPOE50"],
    )
    rooftop = numeric(rooftop, ["POWERPOE50"])
    rooftop["version_time"] = pd.to_datetime(rooftop["VERSION_DATETIME"])
    rooftop["valid_time"] = pd.to_datetime(rooftop["INTERVAL_DATETIME"])
    rooftop = rooftop.loc[rooftop["REGIONID"].isin(REGIONS)]

    if origin_start_day < 0 or origin_end_buffer_days < 0:
        raise ValueError("origin day controls must be non-negative")
    last_origin = month_end - pd.Timedelta(days=max(1, origin_end_buffer_days))
    origins = pd.date_range(
        month_start.normalize() + pd.Timedelta(days=origin_start_day, hours=forecast_origin_hour),
        last_origin.normalize() + pd.Timedelta(hours=forecast_origin_hour),
        freq="1D",
    )
    outputs: list[pd.DataFrame] = []
    for origin in origins:
        issued = latest_at_origin(stpasa, origin, "run_time")
        rooftop_issued = latest_at_origin(rooftop, origin, "version_time")
        issued = issued.merge(
            rooftop_issued[["REGIONID", "valid_time", "POWERPOE50", "version_time"]],
            on=["REGIONID", "valid_time"],
            how="left",
        )
        issued["POWERPOE50"] = issued["POWERPOE50"].fillna(0.0)
        issued["forecast_origin"] = origin
        issued["lead_hours"] = (issued["valid_time"] - origin).dt.total_seconds() / 3600.0
        issued["load"] = issued["DEMAND50"] + issued["POWERPOE50"]
        issued["wind"] = issued["SS_WIND_UIGF"]
        issued["pv"] = issued["SS_SOLAR_UIGF"] + issued["POWERPOE50"]
        issued["net_load"] = issued["load"] - issued["wind"] - issued["pv"]
        outputs.append(
            issued[
                [
                    "forecast_origin",
                    "valid_time",
                    "lead_hours",
                    "REGIONID",
                    "load",
                    "wind",
                    "pv",
                    "net_load",
                    "run_time",
                    "version_time",
                ]
            ]
        )
    forecasts = pd.concat(outputs, ignore_index=True)
    actual_targets = actuals.rename(columns={"timestamp": "valid_time"})[
        ["valid_time", "REGIONID", "load", "wind", "pv", "net_load"]
    ]
    return forecasts.merge(
        actual_targets,
        on=["valid_time", "REGIONID"],
        how="left",
        suffixes=("_forecast", "_actual"),
    )


def audit(actuals: pd.DataFrame, forecasts: pd.DataFrame, capacity: pd.DataFrame) -> dict:
    task_columns = ["load", "wind", "pv", "net_load"]
    expected_actual_rows = int(actuals["timestamp"].nunique() * len(REGIONS))
    horizon_rows = {}
    official_mae = {}
    for days in (1, 2, 3, 7):
        subset = forecasts.loc[forecasts["lead_hours"].le(24 * days)]
        horizon_rows[str(days)] = int(len(subset))
        official_mae[str(days)] = {
            task: float(np.nanmean(np.abs(subset[f"{task}_forecast"] - subset[f"{task}_actual"])))
            for task in task_columns
        }
    return {
        "regions": list(REGIONS),
        "actual_rows": int(len(actuals)),
        "expected_actual_rows": expected_actual_rows,
        "actual_completeness": float(len(actuals) / expected_actual_rows),
        "forecast_origins": int(forecasts["forecast_origin"].nunique()),
        "forecast_rows": int(len(forecasts)),
        "horizon_rows": horizon_rows,
        "official_mae_mw": official_mae,
        "negative_net_load_rows": int(actuals["net_load"].lt(0).sum()),
        "generation_imputed_rows": int(actuals.get("generation_imputed", False).sum()),
        "capacity_mw": capacity.to_dict(orient="records"),
        "target_definition": {
            "load": "operational demand plus rooftop PV",
            "wind": "SCADA output from wind DUIDs",
            "pv": "SCADA utility PV plus rooftop PV",
            "net_load": "load minus wind minus PV",
        },
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    month_start = pd.Timestamp(f"{args.month}-01 00:00:00")
    month_end = month_start + pd.offsets.MonthBegin(1)

    actuals, capacity = build_actuals(raw_dir, month_start, month_end)
    forecasts = build_issued_forecasts(
        raw_dir,
        actuals,
        month_start,
        month_end,
        args.forecast_origin_hour,
        origin_start_day=args.origin_start_day,
        origin_end_buffer_days=args.origin_end_buffer_days,
    )
    report = audit(actuals, forecasts, capacity)

    actuals.to_csv(output_dir / "aemo_unified_actuals.csv", index=False)
    forecasts.to_csv(output_dir / "aemo_unified_issued_forecasts.csv", index=False)
    capacity.to_csv(output_dir / "aemo_unified_capacity.csv", index=False)
    (output_dir / "aemo_unified_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
