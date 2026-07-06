"""
MIC regression -- predict the *level* of resistance, not just resistant/susceptible.

Binary R/S throws away most of the phenotype: an isolate just over the breakpoint
and one resistant to 64x the breakpoint are both "R". The CRyPTIC reuse table
records the actual minimum inhibitory concentration (MIC) from broth microdilution
on a doubling-dilution plate, so we can regress the real quantity.

Target: log2(MIC). A doubling-dilution series (0.25, 0.5, 1, 2, 4, ...) becomes
evenly spaced in log2, so a squared-error regressor operates on the natural scale
and "within 1 dilution" == "within 1.0 in log2".

Censoring: plate reads off the ends are recorded as "<=x" (below the lowest tested
dilution) or ">x" (above the highest). We keep them, nudged one dilution beyond the
bound -- crude but standard for a first model, and better than discarding the very
isolates that carry the strongest phenotype.

Metrics (per drug, 5-fold CV, out-of-fold):
  * EA(+/-1)  essential agreement: predicted MIC within one doubling dilution of
              the measured MIC -- the field-standard accuracy measure.
  * RMSE      root-mean-square error in log2 (dilutions).
  * Pearson r correlation of predicted vs measured log2 MIC.
XGBoost is compared against a sparse Ridge baseline.
"""
from __future__ import annotations

import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.features.build_features import load_raw, mutation_matrix
from src.data.download import DRUG_NAMES, _find_id_column

REUSE_DEFAULT = os.path.join("data", "processed", "CRyPTIC_reuse_table_20221019.csv")
_NAME_TO_CODE = {v: k for k, v in DRUG_NAMES.items()}


def parse_mic(s) -> float | None:
    """MIC string -> log2(concentration), censored reads nudged one dilution out."""
    s = str(s).strip()
    m = re.match(r"^(<=|>=|<|>)?\s*([0-9]*\.?[0-9]+)$", s)
    if not m:
        return None
    op, val = m.group(1) or "", float(m.group(2))
    if val <= 0:
        return None
    log2 = np.log2(val)
    if op in (">", ">="):
        log2 += 1.0
    elif op in ("<", "<="):
        log2 -= 1.0
    return float(log2)


def _xgbreg() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", n_jobs=-1, tree_method="hist",
    )


def _eval_drug(X_full: pd.DataFrame, y_log2: pd.Series, drug: str,
               k: int, seed: int) -> dict | None:
    y = y_log2.dropna()
    common = X_full.index.intersection(y.index)
    if len(common) < 50:
        return None
    X, y = X_full.loc[common], y.loc[common]
    if y.nunique() < 3:
        return None

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    oof_b = np.full(len(y), np.nan)
    yv = y.values
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y.iloc[tr]
        m = _xgbreg()
        m.fit(X_tr, y_tr)
        oof[te] = m.predict(X_te)
        rb = Ridge(alpha=1.0)
        rb.fit(csr_matrix(X_tr.values), y_tr)
        oof_b[te] = rb.predict(csr_matrix(X_te.values))

    err = np.abs(oof - yv)
    return {
        "Drug": drug, "n": len(y),
        "EA±1_%": round(100 * float((err <= 1.0).mean()), 1),
        "EA±2_%": round(100 * float((err <= 2.0).mean()), 1),
        "RMSE_log2": round(float(np.sqrt(np.mean((oof - yv) ** 2))), 3),
        "Pearson_r": round(float(pearsonr(oof, yv)[0]), 3),
        "RMSE_ridge": round(float(np.sqrt(np.mean((oof_b - yv) ** 2))), 3),
        "MIC_range_log2": f"{yv.min():.0f}..{yv.max():.0f}",
    }


def run(data_dir: str, reuse_table: str, quality: str = "HIGH",
        k: int = 5, seed: int = 42) -> pd.DataFrame:
    raw = load_raw(data_dir)
    X_full = mutation_matrix(raw["variants"])          # honours TBAMR_* env vars

    ref = pd.read_csv(reuse_table, low_memory=False)
    id_col = _find_id_column(ref)
    ref = ref.set_index(id_col)

    rows = []
    for drug in sorted(_NAME_TO_CODE):
        code = _NAME_TO_CODE[drug]
        mic_col = f"{code}_MIC"
        q_col = f"{code}_PHENOTYPE_QUALITY"
        if mic_col not in ref.columns:
            continue
        sub = ref[mic_col]
        if quality and q_col in ref.columns:
            sub = sub[ref[q_col].astype(str).str.upper() == quality.upper()]
        y_log2 = sub.map(parse_mic).dropna()
        res = _eval_drug(X_full, y_log2, drug, k, seed)
        if res is None:
            print(f"  {drug:14s} skipped (too few usable MICs)")
        else:
            rows.append(res)
    return (pd.DataFrame(rows).sort_values("EA±1_%", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="MIC (log2) regression per drug")
    ap.add_argument("--data", default=os.path.join("data", "vcf"))
    ap.add_argument("--reuse-table", default=REUSE_DEFAULT)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--quality", default="HIGH")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 78)
    print(f"MIC REGRESSION  |  target=log2(MIC), {args.folds}-fold CV, "
          f"quality={args.quality}")
    print("=" * 78)
    df = run(args.data, args.reuse_table, args.quality, args.folds, args.seed)

    os.makedirs(args.reports, exist_ok=True)
    out = os.path.join(args.reports, "mic_regression.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + df.to_string(index=False))
    print(f"\nSaved -> {out}")
    print("\nReading guide:")
    print("  EA±1_%   = essential agreement: predicted MIC within 1 doubling "
          "dilution of measured (the standard MIC-accuracy metric)")
    print("  RMSE_log2 = error in dilutions; Pearson_r = predicted vs measured")
    print("  RMSE_ridge = sparse linear baseline for comparison")


if __name__ == "__main__":
    main()
