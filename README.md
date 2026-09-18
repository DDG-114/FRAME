# FRAME

## Multisource research branch

The current multisource model and PV context-control experiments are in
[`research/multisource`](research/multisource/README.md). The root-level implementation
below remains the earlier multiyear-suite release.

**Forecast-Origin Representation-Controlled Adaptation for Multi-Task Energy Forecasting**

FRAME forecasts load, wind generation, photovoltaic generation and net load with a shared numerical backbone. Frozen Qwen representations, numerical context and availability information control low-rank residual adapters. A conditional decoder produces point forecasts and ordered marginal quantiles, followed by optional center correction and conformal calibration.

## Implementation

- `code/src/energyca_paper/model.py`: shared backbone, context encoder, residual adapters and prediction heads.
- `code/src/energyca_paper/training.py`: joint training and forecast evaluation.
- `code/src/energy_context/`: context construction and cached language-model representations.
- `run_suite.py`: data preparation, context caching, training and calibration.
- `shared_input_contract.py`: synchronized four-task input construction.
- `multiyear_data.py` and `split_plan.json`: AEMO data loading and chronological partitions.
- `code/tests/`: model-control and forecasting-protocol tests.

The implementation retains the internal `energyca_paper` package name for compatibility with existing checkpoints. This release contains the FRAME implementation from the September 16, 2026 multiyear experiment suite; it does not contain experimental results or model weights.

## Installation

Use Python 3.11 or 3.12. Install a CUDA-enabled PyTorch build appropriate for your GPU, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Data

Obtain AEMO data from its official distribution and prepare the following local files. Data and Qwen weights are stored separately from this repository.

```text
AEMO_DATA/
  dataset/
    actuals_5min.parquet
    actuals_30min.parquet
    issued_30min_YYYY-MM.parquet
    official_2h_inputs/
      aligned_inputs_2h_YYYY-MM.parquet
  data/native5min/YYYY-MM/capacity.csv
```

Actuals contain `timestamp`, `REGIONID`, `load`, `wind`, `pv` and `net_load`. Issued forecasts contain `forecast_origin`, `valid_time`, `REGIONID` (or `region`) and the four task values. The 30-minute files use `available_at`; the two-hour files use per-task `<task>_available` and `<task>_available_at` fields. See `multiyear_data.py` for capacity fields, normalization and availability handling.

Horizons are 2 hours at 5-minute resolution and 24/168 hours at 30-minute resolution. Historical windows span seven days. Regions are NSW1, QLD1, VIC1, SA1 and TAS1. Exact chronological boundaries are in `split_plan.json`; training, validation, calibration, test and confirmation are separate partitions.

## GPU workflow

Run commands from the repository root:

```bash
export AEMO_MULTYEAR_DATA_DIR=/path/to/AEMO_DATA
export QWEN_MODEL_PATH=/path/to/Qwen3-8B
export CUDA_VISIBLE_DEVICES=0

python run_suite.py prepare --mode monthly_quick
python run_suite.py cache --mode monthly_quick
python run_suite.py train --mode monthly_quick --variant full --seed 42
```

`monthly_quick` selects a fixed subset of days for faster experiments. For the full protocol, repeat the preparation, cache and training commands with `--mode full`. Model training uses CUDA; context embeddings are computed once with frozen Qwen3-8B and reused.

Training saves checkpoints, metrics and per-origin predictions under `results/<mode>/`. The full training command also applies calibration after evaluation. Calibration can be rerun with:

```bash
python run_suite.py recalibrate --mode full --variant full --seed 42
```

## Ablations

Use the same prepared data and seed with `--variant no_context`, `no_adapters`, `no_rank_gate`, `no_strength_gate` or `no_anchor`. Each ablation is retrained. Run three seeds separately for a repeated-run comparison; select models using validation data and fit post-training corrections on the calibration split.
