Carbapenemase Prediction in Pseudomonas aeruginosa Using Machine Learning on Disk Diffusion Data

This repository contains the Python code used for machine learning-based prediction of carbapenemase production in Pseudomonas aeruginosa based on antimicrobial disk diffusion data.

The analysis uses CatBoost classification models trained on inhibition zone diameters from antimicrobial susceptibility testing and evaluates their performance in an internal test set and two independent external validation datasets.

## Requirements
The analysis requires Python 3 and the following Python packages:
pandas
numpy
scikit-learn
catboost
openpyxl

## Input data
The script expects an Excel file named: data.xlsx
in the same directory as the Python script.

The Excel workbook must contain the following worksheets:
Internal Dataset, External_Centre2, External_Centre3

Internal Dataset is used for model development and internal evaluation. External_Centre2 and External_Centre3 are used as independent external validation datasets.

The column "Carbapenemase" contains the classification target. Isolates labelled "negative" are classified as non-carbapenemase-producing; all other entries are classified as carbapenemase-producing.

The input variables are inhibition zone diameters obtained by disk diffusion testing.

##Feature sets
Four predefined antimicrobial feature sets are evaluated.
complete-panel antibiotics:
PIP, TZP, TAZ, FEP, ATZ, IMI, MEM, CIP, AK, CET, CZA, CID, AZA, IMR, MEV, FPE
minimal-panel:
FEP, ATZ, IMI, MEM, CET, IMR
Standard-panel:
PIP, TZP, TAZ, FEP, ATZ, IMI, MEM, CIP, AK
novel-β-lactam-panel:
CET, CZA, CID, AZA, IMR, MEV, FPE

##Model development
For each feature set, the internal dataset is stratified into:
70% training set
15% validation set
15% internal test set

A CatBoost classifier is trained on the training set. The validation set is used for model selection and early stopping.
The classification threshold is determined using the internal test set with a target sensitivity of 0.95. Once determined, this threshold is kept fixed for evaluation of both external validation datasets.

The external datasets are not used for model training, model selection, or threshold optimization.

##Performance evaluation
Model performance is assessed using:
Area under the receiver operating characteristic curve (AUROC)
Sensitivity
Specificity

For sensitivity and specificity, predictions are classified using the fixed threshold derived from the internal test set.
Confidence intervals are estimated using 1,000 bootstrap iterations and reported as 95% confidence intervals.

##Output
Results are written to the directory:
model_output/

The output includes the trained CatBoost models, classification thresholds, and performance results for the internal and external validation datasets.

##Reproducibility
Random seeds are fixed where applicable to ensure reproducible data splitting, model training, and bootstrap analyses.

##Citation
If you use this code or the associated methodology, please cite the corresponding publication.
Citation details will be added following publication.
