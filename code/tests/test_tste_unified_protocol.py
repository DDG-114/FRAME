from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.evaluate_aemo_official_baseline import match_prediction
from scripts.audit_aaai_exact_aemo_issued_protocol import audit_issued_frame
from scripts.evaluate_perform_official_baseline import select_origin_rows
from scripts.build_aemo_archive import archive_url
from scripts.run_tste_conformal_calibration import bin_masks
from scripts.aggregate_aemo_formal_suite import sharing_run_dir
from scripts.run_tste_point_baselines import issued_target_from_batch as issued_point_target
from scripts.run_tste_probability_baselines import (
    issued_target_from_batch as issued_quantile_target,
    load_frozen_model,
    make_model,
    model_quantiles,
)
from src.energyca_paper.config import aemo_task_specs, perform_task_specs
from src.energyca_paper.metrics import DEFAULT_QUANTILE_LEVELS
from src.data.load_aemo_unified import _complete_issued_origins, _market_notice_contexts
from src.energy_context.data import _latest_issued_block, _prepare_versioned_exogenous_lookups


def test_probability_baseline_models_are_monotone_and_shape_matched() -> None:
    spec = perform_task_specs(history_days=7, horizon_days=2)["load"]
    history = torch.randn(2, spec.history_len)
    for method in (
        "dlinear_quantile",
        "lstm_quantile",
        "patchtst_quantile",
        "dlinear_issued_quantile",
        "lstm_issued_quantile",
        "patchtst_issued_quantile",
    ):
        model = make_model(method, spec, len(DEFAULT_QUANTILE_LEVELS))
        output = model(history)
        assert output.shape == (2, spec.horizon, len(DEFAULT_QUANTILE_LEVELS))
        assert torch.all(output[..., 1:] >= output[..., :-1])


def test_market_notice_context_uses_region_and_forecast_origin_cutoff(tmp_path: Path) -> None:
    notices = pd.DataFrame(
        {
            "available_at": [
                "2024-01-02 00:00:00",
                "2024-01-03 00:00:00",
                "2024-01-04 01:00:00",
            ],
            "regions": ["NSW1", "QLD1", "NSW1"],
            "type_id": ["RESERVE NOTICE", "RESERVE NOTICE", "RESERVE NOTICE"],
            "reason": ["NSW available", "QLD only", "future NSW"],
            "external_reference": ["", "", ""],
        }
    )
    path = tmp_path / "notices.csv"
    notices.to_csv(path, index=False)
    contexts = _market_notice_contexts(
        str(path),
        origins=pd.DatetimeIndex(["2024-01-04 00:00:00"]),
        regions=["NSW1", "QLD1"],
    )
    nsw = contexts[(pd.Timestamp("2024-01-04 00:00:00"), "NSW1")]
    qld = contexts[(pd.Timestamp("2024-01-04 00:00:00"), "QLD1")]
    assert "NSW available" in nsw
    assert "future NSW" not in nsw
    assert "QLD only" not in nsw
    assert "QLD only" in qld


def test_probability_baseline_checkpoint_rebuild_verifies_method_and_spec(
    tmp_path: Path,
) -> None:
    spec = perform_task_specs(history_days=7, horizon_days=2)["load"]
    method = "dlinear_issued_quantile"
    model = make_model(method, spec, len(DEFAULT_QUANTILE_LEVELS))
    checkpoint = tmp_path / "best_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "method": method,
            "spec": spec.to_dict(),
            "best_val_pinball": 0.1,
        },
        checkpoint,
    )
    loaded = load_frozen_model(
        method,
        SimpleNamespace(spec=spec),
        checkpoint=checkpoint,
        device=torch.device("cpu"),
        quantile_count=len(DEFAULT_QUANTILE_LEVELS),
    )
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)

    wrong_method = "lstm_issued_quantile"
    with pytest.raises(ValueError, match="Checkpoint method mismatch"):
        load_frozen_model(
            wrong_method,
            SimpleNamespace(spec=spec),
            checkpoint=checkpoint,
            device=torch.device("cpu"),
            quantile_count=len(DEFAULT_QUANTILE_LEVELS),
        )


def test_issued_residual_models_use_only_the_target_specific_issued_token() -> None:
    spec = aemo_task_specs(history_days=7, horizon_days=1)["pv"]
    tokens = torch.zeros(2, spec.history_len + spec.horizon, 9)
    issued = torch.linspace(0.1, 0.9, spec.horizon)
    tokens[:, spec.history_len :, 5 + spec.task_id] = issued
    batch = {"tokens": tokens}
    expected = issued.repeat(2, 1)
    assert torch.equal(issued_point_target(batch, spec), expected)
    assert torch.equal(issued_quantile_target(batch, spec), expected)

    class ZeroQuantiles(torch.nn.Module):
        def forward(self, history: torch.Tensor) -> torch.Tensor:
            return torch.zeros(history.shape[0], spec.horizon, len(DEFAULT_QUANTILE_LEVELS))

    output = model_quantiles(
        ZeroQuantiles(),
        batch,
        spec=spec,
        method="dlinear_issued_quantile",
        aemo_capacity_factor=True,
    )
    assert torch.equal(output, expected.unsqueeze(-1).expand_as(output))


def test_sharing_layout_keeps_the_same_task_horizon_artifact_name() -> None:
    root = Path("outputs/aemo_formal_suite")
    assert sharing_run_dir(root, "fully_shared", "load", 7, 42).name == (
        "formal_qwen_issued_load-wind-pv-net_load_H1-2-3-7_L7d_seen-regions_seed42"
    )
    assert sharing_run_dir(root, "task_specific", "pv", 2, 2027).name == (
        "formal_qwen_issued_pv_H1-2-3-7_L7d_seen-regions_seed2027"
    )
    assert sharing_run_dir(root, "horizon_specific", "wind", 3, 3407).name == (
        "formal_qwen_issued_load-wind-pv-net_load_H3_L7d_seen-regions_seed3407"
    )
    assert sharing_run_dir(root, "independent", "net_load", 1, 42).name == (
        "formal_qwen_issued_net_load_H1_L7d_seen-regions_seed42"
    )


def test_protocol_specific_lead_bins_use_physical_hours() -> None:
    aemo = aemo_task_specs(history_days=7, horizon_days=7)["wind"]
    perform = perform_task_specs(history_days=7, horizon_days=2)["wind"]
    assert [label for label, _ in bin_masks(aemo.horizon, aemo.cadence_minutes)] == [
        "0-24h", "24-48h", "48-72h", "72-168h"
    ]
    assert [label for label, _ in bin_masks(perform.horizon, perform.cadence_minutes)] == [
        "0-24h", "24-48h"
    ]


def test_perform_official_matching_selects_the_earliest_valid_horizon() -> None:
    origin = pd.Timestamp("2018-01-01 13:00:00")
    frame = pd.DataFrame(
        {
            "forecast_origin": [origin] * 4,
            "valid_time": pd.to_datetime(
                ["2018-01-02 02:00", "2018-01-02 00:00", "2018-01-02 03:00", "2018-01-02 01:00"]
            ),
            "deterministic": [2.0, 0.0, 3.0, 1.0],
        }
    )
    selected = select_origin_rows(frame, origin, 3)
    np.testing.assert_array_equal(selected["deterministic"].to_numpy(), np.asarray([0.0, 1.0, 2.0]))


def test_aemo_official_matching_uses_latest_vintage_available_by_origin() -> None:
    issued = pd.DataFrame(
        {
            "forecast_origin": pd.to_datetime(["2024-01-31 00:00", "2024-02-01 00:00"]),
            "valid_time": pd.to_datetime(["2024-02-01 00:30", "2024-02-01 01:00"]),
            "REGIONID": ["NSW1", "NSW1"],
            "load_forecast": [10.0, 11.0],
        }
    )
    prediction = match_prediction(
        issued,
        task="load",
        origins=np.asarray(["2024-02-01T00:00:00"]),
        nodes=np.asarray(["NSW1"]),
        horizon=2,
        vintage_cache={},
    )
    np.testing.assert_array_equal(prediction, np.asarray([[10.0, 11.0]], dtype=np.float32))


def test_aemo_archive_filename_switch_is_versioned() -> None:
    july = archive_url(pd.Timestamp("2024-07-01"), "DISPATCH_UNIT_SCADA")
    august = archive_url(pd.Timestamp("2024-08-01"), "DISPATCH_UNIT_SCADA")
    assert "PUBLIC_DVD_DISPATCH_UNIT_SCADA_202407010000.zip" in july
    assert "PUBLIC_ARCHIVE%23DISPATCH_UNIT_SCADA%23FILE01%23202408010000.zip" in august


def test_aemo_origin_filter_requires_every_region_and_all_leads() -> None:
    origins = pd.to_datetime(["2024-01-01 00:00", "2024-01-02 00:00"])
    rows = []
    for origin in origins:
        for region in ("NSW1", "VIC1"):
            for lead in range(1, 5):
                if origin == origins[0] and region == "VIC1" and lead == 1:
                    continue
                rows.append(
                    {
                        "forecast_origin": origin,
                        "valid_time": origin + pd.Timedelta(minutes=30 * lead),
                        "REGIONID": region,
                        "load_forecast": 1.0,
                        "wind_forecast": 1.0,
                        "pv_forecast": 1.0,
                        "net_load_forecast": 1.0,
                    }
                )
    eligible = _complete_issued_origins(pd.DataFrame(rows), regions=["NSW1", "VIC1"], horizon_steps=4)
    assert list(eligible) == [origins[1]]


def test_aemo_issued_protocol_audit_rejects_a_future_vintage() -> None:
    origin = pd.Timestamp("2024-01-02 00:00:00")
    rows = []
    for region in ("NSW1", "VIC1"):
        for lead in range(1, 5):
            valid_time = origin + pd.Timedelta(minutes=30 * lead)
            rows.append(
                {
                    "forecast_origin": origin,
                    "valid_time": valid_time,
                    "lead_hours": lead / 2.0,
                    "REGIONID": region,
                    "load_forecast": 10.0,
                    "wind_forecast": 2.0,
                    "pv_forecast": 1.0,
                    "net_load_forecast": 7.0,
                    "run_time": origin - pd.Timedelta(hours=12),
                    "version_time": origin,
                }
            )
    frame = pd.DataFrame(rows)
    valid = audit_issued_frame(
        frame,
        expected_regions=("NSW1", "VIC1"),
        horizon_steps=4,
    )
    assert valid["checks"]["stpasa_run_available_by_origin"]
    assert valid["checks"]["rooftop_version_available_or_absent_by_origin"]
    assert valid["checks"]["unique_origin_region_lead_rows"]

    frame.loc[0, "run_time"] = origin + pd.Timedelta(minutes=1)
    invalid = audit_issued_frame(
        frame,
        expected_regions=("NSW1", "VIC1"),
        horizon_steps=4,
    )
    assert invalid["checks"]["stpasa_run_available_by_origin"] is False


def test_versioned_lookup_matches_full_latest_vintage_scan() -> None:
    origin = pd.Timestamp("2024-02-01 00:00")
    future = pd.date_range(origin + pd.Timedelta(minutes=30), periods=4, freq="30min")
    rows = []
    for issue, shift in ((origin - pd.Timedelta(days=1), 0.0), (origin, 10.0)):
        for lead, valid_time in enumerate(future, start=1):
            rows.append(
                {
                    "forecast_origin": issue,
                    "valid_time": valid_time,
                    "node_id": "NSW1",
                    "issued_load": shift + lead,
                }
            )
    frame = pd.DataFrame(rows)
    expected, expected_coverage, expected_origin = _latest_issued_block(
        frame,
        forecast_origin=origin,
        future_timestamps=future,
        node_id="NSW1",
        columns=["issued_load"],
    )
    data = {
        "issued_exogenous": frame,
        "issued_exogenous_columns": ["issued_load"],
    }
    _prepare_versioned_exogenous_lookups(data, "issued")
    actual, actual_coverage, actual_origin = _latest_issued_block(
        frame,
        forecast_origin=origin,
        future_timestamps=future,
        node_id="NSW1",
        columns=["issued_load"],
        lookup=data["_issued_exogenous_lookup"],
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual_coverage == expected_coverage == 1.0
    assert actual_origin == expected_origin == origin
