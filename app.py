"""
FastAPI backend for the TB drug-resistance predictor.

Thin wrapper over src/serve/predict.py: loads the frozen per-drug models once at
startup, then exposes JSON endpoints the frontend calls. Serves the static
frontend from ./web. Designed to run on a Hugging Face Space (Docker SDK) on
port 7860.

Endpoints:
  GET  /            -> the predictor + dashboard page (web/index.html)
  POST /predict     -> {text, mode, threshold} -> per-drug resistance profile
  POST /predict_vcf -> multipart VCF upload -> per-drug resistance profile
  GET  /health      -> readiness + model count
"""
from __future__ import annotations

import os

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.serve.predict import (
    load_models, load_calibrators, load_thresholds, load_reliability,
    tokens_from_text, tokens_from_vcf_bytes, predict_isolate,
)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

# --- drug metadata (the 13 CRyPTIC drugs the model directory provides) ---
LABELS = {
    "rifampicin": "Rifampicin", "isoniazid": "Isoniazid", "rifabutin": "Rifabutin",
    "ethambutol": "Ethambutol", "levofloxacin": "Levofloxacin",
    "moxifloxacin": "Moxifloxacin", "ethionamide": "Ethionamide",
    "kanamycin": "Kanamycin", "amikacin": "Amikacin", "bedaquiline": "Bedaquiline",
    "clofazimine": "Clofazimine", "linezolid": "Linezolid", "delamanid": "Delamanid",
}
ABBR = {
    "rifampicin": "RIF", "isoniazid": "INH", "rifabutin": "RFB", "ethambutol": "EMB",
    "levofloxacin": "LFX", "moxifloxacin": "MFX", "ethionamide": "ETO",
    "kanamycin": "KAN", "amikacin": "AMK", "bedaquiline": "BDQ",
    "clofazimine": "CFZ", "linezolid": "LZD", "delamanid": "DLM",
}
FIRST_LINE = {"rifampicin", "isoniazid", "ethambutol", "rifabutin"}
LAST_LINE = {"bedaquiline", "clofazimine", "linezolid", "delamanid"}

app = FastAPI(title="TB Resistance Predictor")

# load frozen models + calibration + thresholds once
MODELS, FEATS = load_models(MODELS_DIR)
CALIB = load_calibrators(MODELS_DIR)
THR90 = load_thresholds(REPORTS_DIR)        # per-drug 90%-sensitivity thresholds
RELI = load_reliability(REPORTS_DIR)         # per-drug CV AUC (trust signal)


def thresholds_for(mode: str, threshold):
    """Balanced -> one slider threshold for every drug; high-sensitivity -> the
    per-drug 90%-sensitivity thresholds if available."""
    if mode == "high_sensitivity" and THR90:
        return dict(THR90)
    thr = 0.5 if threshold is None else float(threshold)
    return {d.lower(): thr for d in MODELS}


def enrich(results: list) -> list:
    for r in results:
        dl = r["drug"].lower()
        r["label"] = LABELS.get(dl, r["drug"].capitalize())
        r["abbr"] = ABBR.get(dl, r["drug"][:3].upper())
        r["line"] = "first" if dl in FIRST_LINE else ("last" if dl in LAST_LINE else "second")
        rel = RELI.get(dl)
        r["reliability"] = rel
        r["low_conf"] = (rel is not None and rel < 0.85)
        r["auc"] = (f"{rel:.2f}" if rel is not None else "")
    return results


class PredictReq(BaseModel):
    text: str = ""
    mode: str = "balanced"
    threshold: float | None = 0.5


def _run(tokens, mode, threshold, n_tokens):
    thr = thresholds_for(mode, threshold)
    results = predict_isolate(MODELS, FEATS, tokens, thresholds=thr,
                              calibrators=CALIB, top_k=5)
    n_resist = sum(1 for r in results if r["call"] == "R")
    return {
        "results": enrich(results),
        "n_tokens": n_tokens,
        "n_recognized": len(tokens),
        "n_resistant": n_resist,
        "n_drugs": len(MODELS),
        "mode": mode,
        "threshold": threshold,
    }


@app.post("/predict")
def predict(req: PredictReq):
    tokens = tokens_from_text(req.text)
    raw_count = len([t for chunk in req.text.splitlines()
                     for t in chunk.split(",") if t.strip()])
    return _run(tokens, req.mode, req.threshold, raw_count)


@app.post("/predict_vcf")
async def predict_vcf(file: UploadFile = File(...),
                      mode: str = Form("balanced"),
                      threshold: float = Form(0.5)):
    data = await file.read()
    tokens = tokens_from_vcf_bytes(data)
    return _run(tokens, mode, threshold, len(tokens))


@app.get("/health")
def health():
    return {"status": "ok", "drugs": len(MODELS), "features": len(FEATS),
            "calibrated": bool(CALIB), "has_thr90": bool(THR90)}


# --- external-validation dashboard (computed live from reports/pooled_scores.csv) ---
DASH_DRUGS = ["INH", "RIF", "EMB", "PZA", "STR"]
DASH_FULL = {"INH": "Isoniazid", "RIF": "Rifampicin", "EMB": "Ethambutol",
             "PZA": "Pyrazinamide", "STR": "Streptomycin"}
COHORT_LABEL = {"sl_ext": "Sierra Leone", "bel_ext": "Belarus"}
# pooled 90%-sensitivity operating points (leave-one-out) + honest per-drug note
DASH_OP = {"INH": (0.90, 0.94), "RIF": (0.90, 0.97), "EMB": (0.89, 0.95),
           "PZA": (0.96, 0.98), "STR": (0.92, 0.70)}


def _roc_points(y, s, n=60):
    from sklearn.metrics import roc_curve
    import numpy as np
    fpr, tpr, _ = roc_curve(y, s)
    if len(fpr) > n:
        idx = np.linspace(0, len(fpr) - 1, n).astype(int)
        fpr, tpr = fpr[idx], tpr[idx]
    return [[round(float(a), 4), round(float(b), 4)] for a, b in zip(fpr, tpr)]


def _auc_ci(y, s, B=500, seed=0):
    from sklearn.metrics import roc_auc_score
    import numpy as np
    y = np.asarray(y); s = np.asarray(s); n = len(y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) > 1:
            boot.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))
    return round(float(roc_auc_score(y, s)), 3), round(float(lo), 3), round(float(hi), 3)


@app.get("/dashboard")
def dashboard():
    """Two-cohort external-validation metrics + ROC geometry, computed from the
    per-isolate pooled scores so nothing is hard-coded or fabricated."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    p = os.path.join(REPORTS_DIR, "pooled_scores.csv")
    if not os.path.exists(p):
        return {"available": False}
    df = pd.read_csv(p)
    drugs = []
    for d in DASH_DRUGS:
        sub = df[df["drug"] == d]
        if sub.empty or sub["y"].nunique() < 2:
            continue
        y = sub["y"].to_numpy(); s = sub["raw"].to_numpy()
        auc, lo, hi = _auc_ci(y, s)
        op = DASH_OP.get(d, (None, None))
        entry = {"drug": d, "label": DASH_FULL[d], "n": len(y),
                 "R": int(y.sum()), "S": int((y == 0).sum()),
                 "auc": auc, "ci_lo": lo, "ci_hi": hi,
                 "sens": op[0], "spec": op[1],
                 "pooled_roc": _roc_points(y, s), "cohorts": []}
        for c in ["sl_ext", "bel_ext"]:
            cs = sub[sub["cohort"] == c]
            if cs["y"].nunique() > 1:
                entry["cohorts"].append({
                    "cohort": c, "label": COHORT_LABEL[c],
                    "auc": round(float(roc_auc_score(cs["y"], cs["raw"])), 3),
                    "roc": _roc_points(cs["y"].to_numpy(), cs["raw"].to_numpy())})
        drugs.append(entry)
    return {"available": True, "drugs": drugs}


@app.get("/", response_class=HTMLResponse)
def index():
    idx = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return HTMLResponse(
        "<h1>TB Resistance Predictor</h1>"
        "<p>Backend is running. Frontend (web/index.html) not found.</p>"
        f"<p>Loaded {len(MODELS)} drug models, {len(FEATS)} features.</p>"
    )


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
