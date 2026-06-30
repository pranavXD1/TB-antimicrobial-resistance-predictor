"""
End-to-end driver: synthetic data -> features -> train -> evaluate -> explain.

Run from the project root:
    python run_pipeline.py                # uses synthetic sample data
    python run_pipeline.py --data data/processed   # once real data is in place

This is the script you run after `src/data/download.py` has produced real CSVs;
just point --data at them.
"""
from __future__ import annotations

import os
import argparse
import yaml

from src.data import synthetic
from src.models.train import train_all
from src.models.evaluate import evaluate_all
from src.interpret.explain import global_importance, save_summary_plot, explain_isolate


def load_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TB-AMR pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default=None, help="override data dir")
    parser.add_argument("--explain-drug", default="Rifampicin")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data or cfg.get("data_dir", "data/sample")
    models_dir = cfg.get("models_dir", "models")
    reports_dir = cfg.get("reports_dir", "reports")
    seed = cfg.get("seed", 42)

    print("=" * 70)
    print("TB-AMR PREDICTOR  |  end-to-end pipeline")
    print("=" * 70)

    # 1. Data (generate synthetic if the chosen dir is empty)
    needed = os.path.join(data_dir, "variants.csv")
    if not os.path.exists(needed):
        print(f"\n[1/4] No data in {data_dir} -> generating synthetic sample")
        synthetic.write_sample(
            data_dir, n_isolates=cfg.get("n_isolates", 4000), seed=seed
        )
    else:
        print(f"\n[1/4] Using data in {data_dir}")

    # 2. Train
    print("\n[2/4] Training baseline + XGBoost per drug")
    train_all(data_dir, models_dir, drugs=cfg.get("drugs"),
              test_size=cfg.get("test_size", 0.25), seed=seed,
              xgb_params=cfg.get("xgb_params"))

    # 3. Evaluate
    print("\n[3/4] Evaluating (clinical metrics; xgb vs logistic baseline)")
    metrics = evaluate_all(models_dir, reports_dir)
    print()
    print(metrics.to_string(index=False))

    # 4. Explain
    drug = args.explain_drug
    print(f"\n[4/4] Interpretability for {drug}")
    print(global_importance(models_dir, drug).to_string(index=False))
    plot = save_summary_plot(models_dir, drug,
                             os.path.join(reports_dir, f"shap_{drug.lower()}.png"))
    print(f"  SHAP summary plot -> {plot}")
    expl = explain_isolate(models_dir, drug)
    print(f"\n  Example decision-support output ({expl['isolate']}):")
    print(f"    {drug}: {expl['call']}  (p={expl['predicted_prob_resistant']})")
    for d in expl["top_drivers"]:
        flag = "present" if d["present"] else "absent "
        print(f"      {d['mutation']:<16} [{flag}]  SHAP {d['shap']:+.3f}")

    print("\nDone. Metrics in reports/metrics.csv, models in models/.")


if __name__ == "__main__":
    main()
