# Multisource FRAME

This source snapshot contains the shared forecasting network, frozen-Qwen source representations, semantic source selection, internal residual adapters, forecast-origin bridge, coherent net-load output and accuracy-weighted residual constraint used in the September 18 experiments.

`year_pv_control_code_20260918/model.py` defines the model. `train.py` trains and evaluates it. The other modules construct source, event and weather inputs. The model supports Full, No Context and Structured Fields through the training arguments.

## Training protocol

Training: 2022–2023. Validation: January–June 2024. Calibration: July–December 2024. Evaluation: 2025. The current context-strength experiments train and select checkpoints on training/validation data only. The 2025 period has been used in earlier project development and is a chronological backtest, not a newly untouched holdout.

The launcher uses all available training origins, three seeds, batch size 48, learning rate 0.0001, weight decay 0.01, EMA 0.999, a 60-epoch cosine schedule, at most 20 epochs and early stopping after five non-improving epochs. Selection uses mean validation NMAE plus mean validation NRMSE across four tasks and three horizons. It uses prepared issued-forecast and context inputs without the newly collected weather cache.

The PV control scale multiplies the context vector entering the internal adapters for PV at 24 h and 168 h. Scale 1 preserves the reference architecture; 0.5 and 0.25 are the retraining candidates. PV at 2 h and the other tasks retain their control formulas. Shared weights still update jointly.

## Prepared inputs

Place the existing prepared experiment inputs under this directory:

```text
year_split_20260918/
  data/h{2,24,168}/{train,val,calibration,test}.npz
  data/h{2,24,168}/protocol.json
  feedback/
  event_inputs/
  aux_oof/
  joint_no_weather_source_tables.pt
```

Data, semantic caches, checkpoints and prediction arrays are stored separately from Git. `prepare_year_split.py` constructs annual partitions from the project's prepared multiyear arrays. `prepare_joint_source_tables.py` constructs the common representation table from source/event embedding caches. The `prepare_data.py` module inside the code snapshot supplies shared constants and loaders; its standalone preparation command belongs to the earlier pilot split, not this annual protocol.

## Run

Install the repository dependencies and use a CUDA-enabled PyTorch environment. From the repository root:

```bash
python research/multisource/year_pv_control_code_20260918/test_pv_control.py
python research/multisource/year_pv_control_code_20260918/test_ablation.py
python research/multisource/year_pv_control_code_20260918/test_reference_skill.py

CUDA_VISIBLE_DEVICES=0 bash research/multisource/run_year_pv_control.sh /path/to/python 0.5 050
CUDA_VISIBLE_DEVICES=1 bash research/multisource/run_year_pv_control.sh /path/to/python 0.25 025
```

Each run saves its argument record, model checkpoint, validation predictions and zero/shuffle inference interventions. Matched retrained No Context is a separate comparator; zeroing a trained Full model is an inference-sensitivity check.
