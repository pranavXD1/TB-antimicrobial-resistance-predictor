"""
Prediction backend for the decision-support app -- pure Python, no UI.

Loads the saved per-drug XGBoost models, reconstructs a single isolate's feature
vector from its variants exactly as training did (SNP presence + optional
per-gene burden), and returns a per-drug resistance profile with the top SHAP
drivers behind each call. Kept UI-free so it can be unit-tested and reused.
"""
from __future__ import annotations

import os
import glob
import tempfile
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from src.features.build_features import _extract_pos, GENE_COORDS
from src.data.vcf_fetch import annotate_token
from src.data.download import parse_vcf

CHROM = "NC_000962.3"


def load_models(models_dir: str = "models") -> tuple[dict, list[str]]:
    """Load every xgb_<drug>.joblib and recover the shared feature list."""
    paths = sorted(glob.glob(os.path.join(models_dir, "xgb_*.joblib")))
    if not paths:
        raise FileNotFoundError(f"No xgb_*.joblib in {models_dir}")
    models = {}
    for p in paths:
        drug = os.path.basename(p)[len("xgb_"):-len(".joblib")]
        models[drug] = joblib.load(p)
    feats = list(next(iter(models.values())).get_booster().feature_names)
    return models, feats


def load_thresholds(reports_dir: str = "reports") -> dict[str, float]:
    """Per-drug 90%-sensitivity thresholds from cv_metrics.csv, if present."""
    p = os.path.join(reports_dir, "cv_metrics.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    if "Drug" not in df or "Thr@90" not in df:
        return {}
    return {str(r["Drug"]).lower(): float(r["Thr@90"]) for _, r in df.iterrows()}


def load_reliability(reports_dir: str = "reports") -> dict[str, float]:
    """Per-drug cross-validated AUC from cv_metrics.csv — used to flag drugs whose
    model is too weak to trust (the low-prevalence last-line drugs)."""
    p = os.path.join(reports_dir, "cv_metrics.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    if "Drug" not in df or "AUC_xgb" not in df:
        return {}
    return {str(r["Drug"]).lower(): float(r["AUC_xgb"]) for _, r in df.iterrows()}


def normalize_token(tok: str) -> str | None:
    """Coerce a pasted token to the annotated 'gene@pos_ref>alt' form used by the
    models. Accepts already-annotated tokens, raw 'CHROM_pos_ref_alt', and bare
    'pos_ref_alt'."""
    tok = tok.strip()
    if not tok:
        return None
    if "@" in tok or (tok[:1] == "g" and tok[1:2].isdigit()):
        return tok
    tail = tok.rsplit("_", 3)                    # [chrom, pos, ref, alt]
    if len(tail) == 4 and tail[1].isdigit():     # CHROM_pos_ref_alt (chrom may hold '_')
        return annotate_token(tok)
    bare = tok.split("_")
    if len(bare) == 3 and bare[0].isdigit():     # pos_ref_alt
        return annotate_token(f"{CHROM}_{tok}")
    return tok


def tokens_from_text(text: str) -> set[str]:
    """Parse a pasted list (newline/comma separated) into annotated tokens."""
    raw = [t for chunk in text.splitlines() for t in chunk.split(",")]
    return {n for n in (normalize_token(t) for t in raw) if n}


def tokens_from_vcf_bytes(data: bytes) -> set[str]:
    """Parse an uploaded VCF (bytes) into annotated tokens, indels included."""
    suffix = ".vcf.gz" if data[:2] == b"\x1f\x8b" else ".vcf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        raw = parse_vcf(path, "input", snps_only=False)
    finally:
        os.unlink(path)
    return {annotate_token(tok) for _, tok in raw}


def build_vector(tokens: set[str], feats: list[str]) -> pd.DataFrame:
    """Reconstruct the model's feature vector for one isolate: SNP presence, plus
    per-gene burden (count of the isolate's variant positions inside each gene)."""
    gene_hits: dict[str, set] = {}
    for t in tokens:
        pos = _extract_pos(t)
        if pos is None:
            continue
        for name, s, e in GENE_COORDS:
            if s <= pos <= e:
                gene_hits.setdefault(name, set()).add(pos)
                break
    row = []
    for f in feats:
        if f.startswith("burden::"):
            row.append(len(gene_hits.get(f.split("::", 1)[1], set())))
        else:
            row.append(1 if f in tokens else 0)
    return pd.DataFrame([row], columns=feats)


def predict_isolate(models: dict, feats: list[str], tokens: set[str],
                    thresholds: dict | None = None, top_k: int = 5) -> list[dict]:
    """Per-drug resistance profile for one isolate, sorted most→least resistant."""
    thresholds = thresholds or {}
    x = build_vector(tokens, feats)
    dm = xgb.DMatrix(x)
    results = []
    for drug, m in models.items():
        prob = float(m.predict_proba(x)[:, 1][0])
        thr = thresholds.get(drug, 0.5)
        contribs = m.get_booster().predict(dm, pred_contribs=True)[0]
        order = np.argsort(-np.abs(contribs[:-1]))
        drivers = [{"feature": feats[i], "contrib": float(contribs[i]),
                    "present": int(x.iloc[0, i])}
                   for i in order[:top_k] if abs(contribs[i]) > 1e-6]
        results.append({"drug": drug, "prob": prob, "threshold": thr,
                        "call": "R" if prob >= thr else "S", "drivers": drivers})
    results.sort(key=lambda r: r["prob"], reverse=True)
    return results
