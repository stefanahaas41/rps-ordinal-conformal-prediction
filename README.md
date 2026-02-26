# rps-ordinal-conformal-prediction

## Overview
This repository implements ranked probability score (RPS)-based ordinal conformal prediction and
runs experiments.

## Contents
- `losses/`: COPOC output layer.
- `scores/`: Conformity scores (MinCPS, Naive CDF, RPS).
- `util/`: Metrics and experiment runner.
- `tabular_data_experiments.ipynb`: Notebook for tabular experiments.
- `eval.ipynb`: Evaluation notebook.

## Setup
Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data
The TOC-UCO dataset is **not** included in this repository.
Obtain it from the original source and place it under `TOC-UCO/` with the same structure.
Source: https://www.uco.es/grupos/ayrna/materials/tocuco/ 

## Running experiments
For notebook-based experiments, open `tabular_data_experiments.ipynb` and run the cells in order.
The helper entry point is `util/tabular_experiments_runner.py`.

## Outputs
Typical outputs include CSV files with metrics and predicted probabilities (e.g., `*_experiments.csv`).

## Evaluation
To evaluate the results, open `eval.ipynb` and run the cells to load the results and compute metrics.

