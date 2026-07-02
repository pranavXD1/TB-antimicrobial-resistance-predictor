"""
Phase 2 -- multi-task resistance model.

ONE network predicts resistance to ALL drugs at once from a shared
representation of the mutation profile. Why this should beat the per-drug
XGBoost baseline (Phase 0):

  * Co-resistance is real -- MDR-TB means rpoB/katG travel together, and shared
    efflux/permeability mechanisms affect several drugs. A shared trunk lets the
    model transfer signal between drugs instead of relearning it 13 times.
  * Data-poor drugs borrow strength from data-rich ones. Second-line drugs with
    few resistant isolates benefit from the representation learned on rifampicin
    and isoniazid.

The key engineering detail is the MASKED loss: CRyPTIC (and the synthetic set)
don't test every isolate against every drug, so the label matrix Y has NaNs. We
compute the loss only over observed (isolate, drug) pairs -- otherwise the model
would train on fabricated labels.

>>> The network trains on YOUR machine (PyTorch + your GPU). <<<
The data assembly, the masked-loss math, and the evaluation/comparison harness
in this file were validated without torch in the dev sandbox; only the
nn.Module training loop needs a torch install (set device='cuda' for the GPU).

Run:
    python -m src.models.multitask                 # synthetic
    python -m src.models.multitask --data data/processed   # real CRyPTIC data
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.features.build_features import label_matrix, mutation_matrix, load_raw
from src.models.evaluate import _metrics

# torch is imported lazily inside the training function so this module can be
# inspected / its non-torch helpers tested without torch installed.


# ---------------------------------------------------------------------------
# Masked binary cross-entropy -- defined in numpy too so the math is testable.
# ---------------------------------------------------------------------------
def masked_bce_numpy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray,
                     pos_weight: np.ndarray | None = None) -> float:
    """Reference implementation (used to validate the torch version's masking)."""
    p = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    w = 1.0 if pos_weight is None else (targets * pos_weight + (1 - targets))
    loss = -(w * (targets * np.log(p) + (1 - targets) * np.log(1 - p)))
    loss = loss * mask
    return float(loss.sum() / max(mask.sum(), 1))


def _pos_weight(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-drug (#negatives / #positives) over observed labels, for imbalance."""
    pos = np.nansum(np.where(mask.astype(bool), Y, np.nan) == 1, axis=0)
    neg = np.nansum(np.where(mask.astype(bool), Y, np.nan) == 0, axis=0)
    return neg / np.clip(pos, 1, None)


# ---------------------------------------------------------------------------
# The network (torch) + training
# ---------------------------------------------------------------------------
def train_multitask(X_tr, Y_tr, M_tr, pos_weight, n_drugs,
                    epochs=250, lr=1e-3, hidden=(256, 128), dropout=0.3,
                    batch_size=256, val_frac=0.15, patience=30,
                    device="cpu", seed=42):
    """Train the shared-trunk multi-task net (mini-batch SGD, early stopping).
    Returns a fn: X -> per-drug probability matrix."""
    import torch
    import torch.nn as nn

    if device == "cuda" and not torch.cuda.is_available():
        print("    [warn] CUDA not available (CPU-only torch build?) -> using CPU")
        device = "cpu"
    torch.manual_seed(seed)
    dev = torch.device(device)

    class MultiTaskNet(nn.Module):
        def __init__(self, n_feat, n_out, hidden, p):
            super().__init__()
            layers, d = [], n_feat
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
                d = h
            self.trunk = nn.Sequential(*layers)
            self.head = nn.Linear(d, n_out)   # one shared trunk, per-drug logits

        def forward(self, x):
            return self.head(self.trunk(x))

    net = MultiTaskNet(X_tr.shape[1], n_drugs, hidden, dropout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    Xt = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    Yt = torch.tensor(np.nan_to_num(Y_tr), dtype=torch.float32, device=dev)
    Mt = torch.tensor(M_tr, dtype=torch.float32, device=dev)
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=dev)

    def masked_loss(logits, yb, mb):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, yb, weight=(yb * pw + (1 - yb)), reduction="none")
        return (bce * mb).sum() / mb.sum().clamp_min(1)

    # train/val split (for early stopping) over the training isolates
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_val = max(1, int(len(X_tr) * val_frac))
    val_idx = torch.tensor(perm[:n_val], device=dev)
    tr_idx = perm[n_val:]

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        epoch_idx = rng.permutation(tr_idx)          # reshuffle once per epoch
        for s in range(0, len(epoch_idx), batch_size):
            b = torch.tensor(epoch_idx[s:s + batch_size], device=dev)
            opt.zero_grad()
            loss = masked_loss(net(Xt[b]), Yt[b], Mt[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = masked_loss(net(Xt[val_idx]), Yt[val_idx], Mt[val_idx]).item()
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    early stop at epoch {ep+1} (best val BCE {best_val:.4f})")
                break
        if (ep + 1) % 50 == 0:
            print(f"    epoch {ep+1:>3}/{epochs}  val masked BCE {vloss:.4f}")

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=dev)
            return torch.sigmoid(net(xt)).cpu().numpy()
    return predict


# ---------------------------------------------------------------------------
# Apples-to-apples comparison vs per-drug XGBoost  (torch-free; fully tested)
# ---------------------------------------------------------------------------
def per_drug_xgb_aucs(X_tr, Y_tr, M_tr, X_te, Y_te, M_te, drugs, seed=42) -> dict:
    """Train one XGBoost per drug on the SAME split; return per-drug test AUC."""
    out = {}
    for j, drug in enumerate(drugs):
        tr = M_tr[:, j].astype(bool)
        te = M_te[:, j].astype(bool)
        ytr = Y_tr[tr, j]
        if ytr.sum() < 10 or (ytr == 0).sum() < 10:
            out[drug] = float("nan")
            continue
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.8, eval_metric="logloss",
                            tree_method="hist", scale_pos_weight=spw, n_jobs=-1)
        clf.fit(X_tr[tr], ytr)
        p = clf.predict_proba(X_te[te])[:, 1]
        out[drug] = _metrics(Y_te[te, j], p)["AUC"]
    return out


def run(data_dir="data/sample", reports_dir="reports", device="cpu",
        test_size=0.25, seed=42, epochs=250, sample_dir="data/sample") -> pd.DataFrame:
    # Make synthetic runs work out of the box; never fabricate into a real dir.
    if not os.path.exists(os.path.join(data_dir, "variants.csv")):
        if os.path.abspath(data_dir) == os.path.abspath(sample_dir):
            from src.data import synthetic
            print(f"  no data in {data_dir} -> generating synthetic sample")
            synthetic.write_sample(data_dir, n_isolates=4000, seed=seed)
        else:
            msg = [f"No variants.csv in '{data_dir}', so there's nothing to train on."]
            if os.path.exists(os.path.join(data_dir, "phenotypes.csv")):
                msg += ["(phenotypes.csv is there — real data downloaded — but the variant",
                        " matrix hasn't been built yet.) Build it first:",
                        f"  python -m src.data.download variants --vcf-dir <folder> --out {data_dir}"]
            else:
                msg += [f"Run the synthetic demo first: python -m src.models.multitask"]
            raise SystemExit("\n".join(msg))

    raw = load_raw(data_dir)
    Xdf = mutation_matrix(raw["variants"])
    X, Y, drugs = label_matrix(data_dir, X_full=Xdf)
    print(f"  multi-task matrix: {X.shape[0]:,} isolates x {X.shape[1]} mutations "
          f"x {len(drugs)} drugs")
    Xv = X.to_numpy(dtype=np.float32)
    Yv = Y.to_numpy(dtype=float)            # NaN where untested
    Mv = (~np.isnan(Yv)).astype(np.float32) # observed-label mask

    idx = np.arange(len(Xv))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    X_tr, X_te = Xv[tr], Xv[te]
    Y_tr, Y_te = Yv[tr], Yv[te]
    M_tr, M_te = Mv[tr], Mv[te]

    pw = _pos_weight(Y_tr, M_tr)

    print("  training multi-task network...")
    predict = train_multitask(X_tr, Y_tr, M_tr, pw, len(drugs),
                              epochs=epochs, device=device, seed=seed)
    P_te = predict(X_te)

    print("  training per-drug XGBoost baselines on the same split...")
    xgb_auc = per_drug_xgb_aucs(X_tr, Y_tr, M_tr, X_te, Y_te, M_te, drugs, seed)

    rows = []
    for j, drug in enumerate(drugs):
        te_obs = M_te[:, j].astype(bool)
        if te_obs.sum() < 5 or len(np.unique(Y_te[te_obs, j])) < 2:
            continue
        mt = _metrics(Y_te[te_obs, j], P_te[te_obs, j])
        rows.append({
            "Drug": drug, "n_test": int(te_obs.sum()), "%R": f"{Y_te[te_obs, j].mean():.0%}",
            "AUC (xgb)": round(xgb_auc.get(drug, float("nan")), 3),
            "AUC (multitask)": round(mt["AUC"], 3),
            "Sens (mt)": round(mt["Sensitivity"], 3),
            "Spec (mt)": round(mt["Specificity"], 3),
        })
    df = pd.DataFrame(rows)
    df["delta"] = (df["AUC (multitask)"] - df["AUC (xgb)"]).round(3)
    df = df.sort_values("AUC (multitask)", ascending=False).reset_index(drop=True)
    os.makedirs(reports_dir, exist_ok=True)
    df.to_csv(os.path.join(reports_dir, "metrics_multitask.csv"), index=False)
    return df


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Train & evaluate the multi-task model")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--device", default="cpu", help="'cuda' to use your GPU")
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()

    df = run(args.data, args.reports, device=args.device, epochs=args.epochs)
    print("\nMulti-task vs per-drug XGBoost (positive delta = multi-task wins):\n")
    print(df.to_string(index=False))
    wins = (df["delta"] > 0).sum()
    print(f"\nMulti-task >= XGBoost on {wins}/{len(df)} drugs; "
          f"mean delta {df['delta'].mean():+.3f} AUC.")


if __name__ == "__main__":
    main()
