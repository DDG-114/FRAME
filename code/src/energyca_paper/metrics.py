"""Point-forecast metrics used by the EnergyCA-LLM paper protocol."""
from __future__ import annotations

import numpy as np


DEFAULT_QUANTILE_LEVELS = (
    0.01,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.975,
    0.99,
)


def point_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute the paper's global point-forecast metric suite."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    absolute_error = np.abs(error)
    mae = float(np.mean(absolute_error))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nmae": float(mae / max(float(np.mean(np.abs(target))), 1e-8)),
        "smape": float(
            100.0
            * np.mean(2.0 * absolute_error / np.maximum(np.abs(target) + np.abs(prediction), 1e-8))
        ),
    }


def mase(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    insample: np.ndarray,
    seasonality: int,
) -> float:
    """Mean absolute scaled error using the in-sample seasonal-naive scale."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    insample = np.asarray(insample, dtype=np.float64).reshape(-1)
    if len(insample) <= seasonality:
        return float("nan")
    scale = float(np.mean(np.abs(insample[seasonality:] - insample[:-seasonality])))
    return float(np.mean(np.abs(prediction - target)) / max(scale, 1e-8))


def probabilistic_metrics(
    quantiles: np.ndarray,
    target: np.ndarray,
    *,
    levels: np.ndarray | tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
) -> dict[str, float]:
    """Evaluate marginal quantile forecasts without assuming a parametric law."""

    quantiles = np.asarray(quantiles, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    if quantiles.shape[:-1] != target.shape or quantiles.shape[-1] != levels.size:
        raise ValueError("quantiles must have shape target.shape + (number of levels,)")
    residual = target[..., None] - quantiles
    pinball_by_level = np.maximum(levels * residual, (levels - 1.0) * residual)
    pinball = float(np.mean(pinball_by_level))
    # The integral of quantile scores is the CRPS. The reported approximation
    # uses the available quantile grid and remains valid for non-Gaussian heads.
    score_by_level = np.mean(pinball_by_level, axis=tuple(range(pinball_by_level.ndim - 1)))
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    crps = float(2.0 * integrate(score_by_level, levels) / max(levels[-1] - levels[0], 1e-8))
    result = {
        "pinball": pinball,
        "crps": crps,
        "ncrps": float(crps / max(float(np.mean(np.abs(target))), 1e-8)),
        "quantile_crossing_rate": float(np.mean(np.diff(quantiles, axis=-1) < -1e-8)),
    }
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    interval_scores: list[np.ndarray] = []
    for coverage in (0.50, 0.80, 0.90, 0.95):
        alpha = 1.0 - coverage
        lower_index = int(np.argmin(np.abs(levels - alpha / 2.0)))
        upper_index = int(np.argmin(np.abs(levels - (1.0 - alpha / 2.0))))
        lower = quantiles[..., lower_index]
        upper = quantiles[..., upper_index]
        inside = (target >= lower) & (target <= upper)
        width = upper - lower
        interval_score = width + (2.0 / alpha) * np.maximum(lower - target, 0.0) + (2.0 / alpha) * np.maximum(target - upper, 0.0)
        interval_scores.append(interval_score)
        suffix = str(int(coverage * 100))
        result[f"picp_{suffix}"] = float(np.mean(inside))
        result[f"ace_{suffix}"] = float(abs(np.mean(inside) - coverage))
        result[f"pinaw_{suffix}"] = float(np.mean(width) / max(float(np.mean(np.abs(target))), 1e-8))
    wis_components = [np.abs(target - quantiles[..., median_index]), *interval_scores]
    wis = float(np.mean(np.stack(wis_components, axis=-1)))
    result["wis"] = wis
    result["nwis"] = float(wis / max(float(np.mean(np.abs(target))), 1e-8))
    return result
