---
title: TB Resistance Predictor
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TB drug-resistance predictor

Genome-based drug-resistance prediction for *Mycobacterium tuberculosis*, trained
on 12,288 CRyPTIC isolates and externally validated on two independent cohorts
(Sierra Leone, Belarus). Paste an isolate's variants (or upload a VCF) and get a
per-drug resistance profile with calibrated probabilities and the SHAP drivers
behind each call.

**Research tool — not a clinical device.** Predictions do not replace phenotypic
drug-susceptibility testing.

## What this Space needs (repo layout)

Push these to the Space (Docker SDK builds automatically):

```
app.py                      # FastAPI backend (this repo)
Dockerfile                  # this repo
requirements.txt            # this repo
web/index.html              # frontend (served at /)
src/                        # the package predict.py lives in
  __init__.py
  serve/predict.py
  data/vcf_fetch.py         # annotate_token  (imported by predict.py)
  data/download.py          # parse_vcf       (imported by predict.py)
  features/build_features.py
models/                     # SLIM: xgb_<drug>.joblib + calibrators.joblib + drugs.json
                            # (NOT the testbundle_*.joblib — evaluation only)
reports/cv_metrics.csv      # optional: enables high-sensitivity thresholds + trust flags
```

The slim `models/` is ~29 MB (inference artifacts only), so it commits directly
via Git LFS — no external download needed at boot. The full `models/` (with
`testbundle_*.joblib`) and the DuckDB stay on Zenodo for reproducibility; the app
does not need them.

## Deploy

```bash
# 1. create a Space at huggingface.co/new-space  (SDK: Docker)
# 2. clone it, copy the files above in, track the models with LFS
git lfs install
git lfs track "models/*.joblib"
git add .gitattributes app.py Dockerfile requirements.txt web src models reports
git commit -m "TB resistance predictor: FastAPI + frozen models"
git push
```

The Space builds the image and serves at the Space URL. Check `/health` for
readiness (returns model + feature counts).

## API

- `POST /predict` — `{ "text": "rpoB@761139_C>T\nkatG@2154724_G>C", "mode": "balanced", "threshold": 0.5 }`
- `POST /predict_vcf` — multipart file upload (`.vcf` / `.vcf.gz`)
- `GET /health` — readiness

Modes: `balanced` (call resistant at the slider threshold, default 0.5) or
`high_sensitivity` (per-drug 90%-sensitivity thresholds from `cv_metrics.csv`).
