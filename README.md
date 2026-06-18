# Ranked Probability Score for Ordinal Conformal Prediction

## Overview

This repository provides the implementation for using the **Ranked Probability Score (RPS)** as a conformity score in conformal prediction for **ordinal classification** tasks. The RPS, introduced by [Epstein (1969)](https://doi.org/10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2), is a proper scoring rule that penalises the cumulative-distribution distance between the forecast and the observation, making it inherently suited for ordered categories.

We compare the RPS-based conformity score against several baselines — APS, LAC, Naive CDF (OCDF), and Min-CPS — across tabular and image benchmarks, using both standard cross-entropy and the unimodal COPOC output layer.

## Repository Structure

```
├── scores/
│   ├── rps.py                          # RPS conformity score (Epstein, 1969)
│   ├── naive.py                        # Naive CDF conformity score (Lu et al., 2022)
│   └── minCPS.py                       # Min-CPS conformity score (Zhang et al., 2026)
├── util/
│   ├── metrics.py                      # Classification & conformal prediction metrics
│   ├── tabular_experiments_runner.py   # Tabular experiment pipeline
│   └── image_experiment_runner.py      # Image experiment pipeline
├── tabular_data_experiments.ipynb      # Tabular experiments notebook
├── eval.ipynb                          # Evaluation & plotting notebook
├── BACH.ipynb                          # BACH histopathology experiment notebook
├── RetinaMNIST.ipynb                   # RetinaMNIST experiment notebook
├── FGNet.ipynb                         # FG-NET age estimation experiment notebook
├── results/                            # Pre-computed experimental results (CSV)
├── requirements.txt                    # Python dependencies
└── README.md
```

## Experiments

### Tabular Experiments

Tabular ordinal classification experiments use datasets from the **TOC-UCO** benchmark collection. The pipeline trains a multi-layer perceptron (with and without the COPOC output layer), then evaluates conformal prediction sets under multiple conformity scores (RPS, APS, LAC, OCDF, Min-CPS) across 50 random calibration/test splits.

**Notebook:** `tabular_data_experiments.ipynb`

### Image Experiments

Image experiments fine-tune a **ResNet-18** (pretrained on ImageNet) on three ordinal image datasets. The same conformal prediction pipeline is applied, comparing all conformity scores with both cross-entropy and COPOC losses.

#### BACH (BreastPathQ)

Histopathology image classification into four ordinal tissue categories (Normal, Benign, *In situ* carcinoma, Invasive carcinoma). The BACH dataset provides microscopy images for breast cancer grading.

**Notebook:** `BACH.ipynb`

#### RetinaMNIST

Retinal fundus images from the [MedMNIST](https://medmnist.com/) collection, classified into five ordinal diabetic retinopathy severity grades (0–4).

**Notebook:** `RetinaMNIST.ipynb`

#### FG-NET

Face images from the FG-NET Aging Database, discretised into ordinal age groups for age estimation as an ordinal classification task.

**Notebook:** `FGNet.ipynb`

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Data

### TOC-UCO (Tabular)

The TOC-UCO tabular datasets are **not** bundled in this repository. Download them from the original source and place them under `TOC-UCO/data/`:

> <https://www.uco.es/grupos/ayrna/materials/tocuco/>

### Image Datasets

- **BACH** – see `BACH.ipynb` for download instructions.
- **RetinaMNIST** – downloaded automatically via the `medmnist` package.
- **FG-NET** – see `FGNet.ipynb` for download instructions.

## Running Experiments

1. **Tabular:** open `tabular_data_experiments.ipynb` and run all cells.
2. **Image:** open the corresponding notebook (`BACH.ipynb`, `RetinaMNIST.ipynb`, or `FGNet.ipynb`) and run all cells.
3. **Evaluation:** open `eval.ipynb` to load result CSVs and reproduce plots.

## Results

Pre-computed experimental results for the paper can be found in the `results/` folder as CSV files. These include conformal prediction results for all datasets and conformity scores reported in the paper.

## References

- Epstein, E. S. (1969). A scoring system for probability forecasts of ranked categories. Journal of Applied Meteorology (1962-1982), 8(6), 985-987.
- Dey, P., Merugu, S., & Kaveri, S. R. (2023). Conformal prediction sets for ordinal classification. Advances in Neural Information Processing Systems, 36, 879-899.
- Lu, C., Angelopoulos, A. N., & Pomerantz, S. (2022, September). Improving trustworthiness of AI disease severity rating in medical imaging with ordinal conformal prediction sets. In International conference on medical image computing and computer-assisted intervention (pp. 545-554). Cham: Springer Nature Switzerland.
- Zhang, Z., Chen, X., Shi, Y., Ma, L. L., Xu, Z., & Yan, Y. (2026, March). Minimum-Length Conformal Prediction Sets for Ordinal Classification. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 40, No. 34, pp. 28662-28670).


