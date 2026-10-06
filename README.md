# DeepBiomarker: Interpretable Deep Learning on Longitudinal EHR

Predicting psychiatric hospitalization from patients' visit histories (diagnoses, medications, abnormal labs,
procedures) and identifying which factors drive risk. A public, synthetic-data reimplementation of the approach
behind the DeepBiomarker research line.

**Status:** in progress. Synthetic data generator with planted risk factors and logistic regression baselines are
complete; sequence model, contribution analysis, and polygenic risk extension are next.

## Why synthetic data with planted risk factors
The outcome is generated from a known set of risk and protective factors, with recent events weighted more than
older ones. Because the truth is known, the project can test not only prediction accuracy but whether a model's
explanations recover the real risk factors.

## Quick start
```bash
pip install -r requirements.txt
pip install -e .
pytest
python -m deepbiomarker.generate
python -m deepbiomarker.baseline
```
