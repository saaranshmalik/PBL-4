## Dataset Training And Testing

This project now includes a standalone dataset training and testing module in [dataset_training_module.py](/e:/codes/pbl-4/PBL-4-main/dataset_training_module.py:1).

### What it produces

- Train and test accuracy
- Train and validation loss curves
- Train and validation accuracy curves
- A confusion matrix graph
- A class distribution graph
- A `metrics.json` file
- Saved model weights in `.npz` format

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run with the default MELD dataset

```powershell
.\run_dataset_training.ps1
```

The runner now defaults to the integrated `MELD` dataset.

### Run with the synthetic dataset

```powershell
.\run_dataset_training.ps1 --dataset synthetic
```

### Run with a CSV dataset

The CSV file must contain:

- Numeric feature columns
- A `label` column with integer values from `0` to `6`

Example:

```powershell
.\run_dataset_training.ps1 --dataset-path .\your_dataset.csv
```

### Output location

Each run creates a timestamped folder under `outputs/` containing the graphs, metrics, and saved weights.
