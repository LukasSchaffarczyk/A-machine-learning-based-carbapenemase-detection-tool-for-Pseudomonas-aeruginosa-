# Carbapenemase Prediction in *Pseudomonas aeruginosa* Using Machine Learning on Disk Diffusion Data

## Overview

This repository contains Python code for machine learning-based prediction of carbapenemase production in *Pseudomonas aeruginosa* using antimicrobial disk diffusion data. CatBoost classification models are trained on inhibition zone diameters (IZDs) and evaluated using an internal dataset and two independent external validation datasets.

## Requirements

Python 3.12 and the following packages are recommended to reproduce the documented software environment:

| Package | Version |
|---|---|
| pandas | 2.2.2 |
| NumPy | 1.26.4 |
| scikit-learn | 1.5.1 |
| CatBoost | 1.2.10 |
| openpyxl | 3.1.5 |

## Installation

### Step 1: Download the repository

Clone the repository with Git:

```bash
git clone https://github.com/LukasSchaffarczyk/A-machine-learning-based-carbapenemase-detection-tool-for-Pseudomonas-aeruginosa-
cd ~/A-machine-learning-based-carbapenemase-detection-tool-for-Pseudomonas-aeruginosa-
```

Alternatively, download the repository as a ZIP archive from GitHub and extract it.

### Step 2: Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

Install the required Python packages:

```bash
python -m pip install pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.1 catboost==1.2.10 openpyxl==3.1.5
```

## Input data

Download the Excel workbook named `data.xlsx` in the same directory as `CarbaDetector_Pseudomonas.py`.

The workbook contain the following worksheets:

| Worksheet | Purpose |
|---|---|
| `Internal Dataset` | Model development and internal evaluation |
| `External_Centre2` | Independent external validation |
| `External_Centre3` | Independent external validation |

The `Carbapenemase` column contains the classification target. Isolates labelled `negative` are classified as non-carbapenemase-producing *P. aeruginosa* (nCPPA); all other entries are classified as carbapenemase-producing *P. aeruginosa* (CPPA).

Predictor variables are inhibition zone diameters from antimicrobial disk diffusion testing. Column names must match the antimicrobial abbreviations used below.

## Feature sets

Four predefined antimicrobial feature sets are evaluated:

| Feature set | Antimicrobial variables |
|---|---|
| Complete panel | PIP, TZP, TAZ, FEP, ATZ, IMI, MEM, CIP, AK, CET, CZA, CID, AZA, IMR, MEV, FPE |
| Minimal panel | FEP, ATZ, IMI, MEM, CET, IMR |
| Standard panel | PIP, TZP, TAZ, FEP, ATZ, IMI, MEM, CIP, AK |
| Novel β-lactam panel | CET, CZA, CID, AZA, IMR, MEV, FPE |

## Running the analysis

From the repository directory, with the virtual environment activated, run:

```bash
python CarbaDetector_Pseudomonas.py
```

## Model development

For each feature set, the internal dataset is stratified into a 70% training set, a 15% validation set, and a 15% internal test set.

A CatBoost classifier is trained using the training set. The validation set is used for model selection and early stopping. The classification threshold is determined using the internal test set with a target sensitivity of 0.95. This threshold is kept fixed when evaluating both external validation datasets.

The external datasets are not used for model training, model selection, or threshold optimization.

## Performance evaluation

Performance is assessed using area under the receiver operating characteristic curve (AUROC), sensitivity, and specificity. Sensitivity and specificity are calculated using the fixed threshold derived from the internal test set. Confidence intervals are estimated from 1,000 bootstrap iterations and reported as 95% confidence intervals.

## Output

Results are written to:

```text
model_output/
```

Outputs include trained CatBoost models, classification thresholds, and performance results for the internal and external validation datasets.

## Reproducibility

Random seeds are fixed where applicable to support reproducible data splitting, model training, and bootstrap analyses. The package versions above document the intended Python environment.

## Citation

If you use this code or the associated methodology, please cite the corresponding publication. Citation details will be added following publication.
