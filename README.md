---
title: TB Drug-Resistance Predictor
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TB Drug-Resistance Predictor

Genome-based resistance prediction for *Mycobacterium tuberculosis*, trained on the
CRyPTIC panel (12,288 isolates) plus GenTB, and externally validated on two
independent cohorts (Sierra Leone, PRJEB7727; Belarus, PRJNA200335; pooled n = 186).

Per-drug XGBoost classifiers over a candidate-gene variant representation, with
calibrated probabilities and per-prediction SHAP attributions. Paste an isolate's
variants or upload a VCF to get a resistance profile across 13 drugs.

**Research tool — not a clinical device.** Predictions are not validated for patient
care; confirm with phenotypic drug-susceptibility testing.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```

## Endpoints

- `GET  /`            — predictor + validation dashboard (web/index.html)
- `POST /predict`     — `{text, mode, threshold}` → per-drug results
- `POST /predict_vcf` — multipart VCF upload → per-drug results
- `GET  /dashboard`   — pooled external-validation metrics (AUC, CIs, ROC points)
- `GET  /health`      — liveness

## Method

See the **Method** tab in the app for the full writeup: training data, feature
representation, model + calibration, the six-step external-validation pipeline,
cohort details, and how to read the metrics.
