#!/usr/bin/env python
"""Download, process, verify, and compact a range of AEMO archive months."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "DISPATCH_UNIT_SCADA",
    "DISPATCHREGIONSUM",
    "DUDETAILSUMMARY",
    "GENUNITS",
    "ROOFTOP_PV_ACTUAL",
    "ROOFTOP_PV_FORECAST",
    "STPASA_REGIONSOLUTION",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_month", required=True, help="YYYY-MM")
    parser.add_argument("--end_month", required=True, help="YYYY-MM, inclusive")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cleanup_raw", action="store_true")
    return parser.parse_args()


def month_range(start: str, end: str) -> list[pd.Timestamp]:
    values = list(pd.date_range(f"{start}-01", f"{end}-01", freq="MS"))
    if not values:
        raise ValueError("The requested AEMO month range is empty")
    return values


def archive_url(month: pd.Timestamp, table: str) -> str:
    year = month.strftime("%Y")
    year_month = month.strftime("%Y_%m")
    stamp = month.strftime("%Y%m010000")
    directory = (
        "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        f"{year}/MMSDM_{year_month}/MMSDM_Historical_Data_SQLLoader/DATA/"
    )
    if month >= pd.Timestamp("2024-08-01"):
        return directory + f"PUBLIC_ARCHIVE%23{table}%23FILE01%23{stamp}.zip"
    return directory + f"PUBLIC_DVD_{table}_{stamp}.zip"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() and destination.stat().st_size > 0:
        try:
            with zipfile.ZipFile(destination) as payload:
                if payload.namelist():
                    return
        except zipfile.BadZipFile:
            destination.replace(partial)
    subprocess.run(
        [
            "curl",
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry", "20",
            "--retry-delay", "3",
            "--retry-max-time", "3600",
            "--retry-all-errors",
            "--connect-timeout", "30",
            "--max-time", "3600",
            "--speed-time", "60",
            "--speed-limit", "1024",
            "--continue-at", "-",
            "--output", str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(destination)
    with zipfile.ZipFile(destination) as payload:
        if not payload.namelist():
            raise ValueError(f"Downloaded AEMO archive is empty: {destination}")


def output_complete(output_dir: Path) -> bool:
    audit_path = output_dir / "aemo_unified_audit.json"
    if not audit_path.exists():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    required = (
        output_dir / "aemo_unified_actuals.csv",
        output_dir / "aemo_unified_issued_forecasts.csv",
        output_dir / "aemo_unified_capacity.csv",
    )
    return (
        all(path.exists() and path.stat().st_size > 0 for path in required)
        and float(audit.get("actual_completeness", 0.0)) == 1.0
        and int(audit.get("forecast_origins", 0)) >= 27
    )


def process_month(month: pd.Timestamp, workspace: Path, cleanup_raw: bool) -> dict:
    label = month.strftime("%Y-%m")
    raw_dir = workspace / "raw" / label
    output_dir = workspace / "processed" / label
    if not output_complete(output_dir):
        for table in TABLES:
            stamp = month.strftime("%Y%m010000")
            destination = raw_dir / f"PUBLIC_DVD_{table}_{stamp}.zip"
            download(archive_url(month, table), destination)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_aemo_unified_poc.py"),
            "--raw_dir", str(raw_dir),
            "--output_dir", str(output_dir),
            "--month", label,
            "--origin_start_day", "0",
            "--origin_end_buffer_days", "0",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not output_complete(output_dir):
        raise RuntimeError(f"AEMO month {label} did not pass the completeness audit")
    if cleanup_raw:
        resolved_root = (workspace / "raw").resolve()
        for table in TABLES:
            target = (raw_dir / f"PUBLIC_DVD_{table}_{month.strftime('%Y%m010000')}.zip").resolve()
            if resolved_root not in target.parents:
                raise RuntimeError(f"Refusing to remove archive outside {resolved_root}: {target}")
            if target.exists():
                target.unlink()
    audit = json.loads((output_dir / "aemo_unified_audit.json").read_text(encoding="utf-8"))
    return {
        "month": label,
        "actual_rows": audit["actual_rows"],
        "forecast_origins": audit["forecast_origins"],
        "forecast_rows": audit["forecast_rows"],
        "negative_net_load_rows": audit["negative_net_load_rows"],
        "raw_archives_removed": bool(cleanup_raw),
    }


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace / "archive_build_manifest.json"
    completed: list[dict] = []
    for month in month_range(args.start_month, args.end_month):
        record = process_month(month, workspace, args.cleanup_raw)
        completed.append(record)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_month": args.start_month,
            "end_month": args.end_month,
            "cleanup_raw": bool(args.cleanup_raw),
            "months": completed,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
