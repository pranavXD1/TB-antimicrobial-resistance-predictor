# TB-AMR Predictor

**Predicting *Mycobacterium tuberculosis* antibiotic resistance from genomic mutations — with explanations a clinician can act on.**

Given the mutations present in a TB isolate's genome, predict whether it is **resistant (R)** or **susceptible (S)** to each drug, and surface *which mutations drove each call*. The goal is a decision-support tool: don't just output a probability, output "resistant to rifampicin — driven by rpoB S450L."

> **Status — Phase 0 complete.** Project scaffold + an end-to-end pipeline
> (data → features → train → evaluate → interpret) that runs today on a
> biologically-realistic **synthetic** dataset. Swap in real CRyPTIC/NCBI data
> (`src/data/download.py`) and the same pipeline runs unchanged.

---

## Why this matters

Tuberculosis is among the world's deadliest infectious diseases, and drug-resistant TB is a top WHO global-health threat. Phenotypic drug-susceptibility testing is slow (weeks of culture); predicting resistance directly from a sequenced genome is fast and increasingly viable. The hard part isn't just accuracy — it's **trust and interpretability**: a clinician needs to know *why* the model called an isolate resistant, and the model must lean on real resistance genes rather than on lineage/background artefacts.

This project is the same modelling muscle as my HOPV molecular work (featurise a biological entity → predict a property → interpret the drivers), pointed at a higher-stakes, differently-shaped problem.

## How it works

```
   genome variant calls                  drug-susceptibility results
   (isolate, mutation)                   (isolate, drug, R/S)
            │                                      │
            ▼                                      ▼
  ┌─────────────────────┐              ┌───────────────────────┐
  │  build_features.py  │   pivot →    │   per-drug label sets  │
  │  binary mutation X  │   align      │   y ∈ {R=1, S=0}       │
  └─────────────────────┘              └───────────────────────┘
            └──────────────────┬───────────────────┘
                               ▼
        ┌───────────────────────────────────────────┐
        │  train.py   logistic baseline  +  XGBoost   │   per drug,
        │             (class-imbalance aware)         │   stratified split
        └───────────────────────────────────────────┘
                               ▼
     ┌──────────────────────┐        ┌─────────────────────────────┐
     │   evaluate.py        │        │   explain.py (SHAP)          │
     │   clinical metrics:  │        │   global: which mutations?   │
     │   sensitivity-first  │        │   local: why THIS isolate?   │
     └──────────────────────┘        └─────────────────────────────┘
```

**Design choices that matter**

- **Source-agnostic data seam.** Modelling code never knows whether the CSVs came from the synthetic generator or the real download — so dev, CI, and production all share one code path.
- **Sensitivity-first evaluation.** Missing a resistant isolate (→ prescribing a useless drug) is worse than a false alarm, so we foreground sensitivity/recall alongside specificity and AUC.
- **Baseline vs. model, always visible.** Every drug is scored with a logistic baseline *and* XGBoost, so model complexity has to earn its keep.
- **Interpretability is a first-class output, not an afterthought** — it's the seed of the clinical decision-support layer.

## Project structure

```
tb-amr-predictor/
├── run_pipeline.py            # end-to-end driver
├── config.yaml                # seed, drugs, paths, model params
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── synthetic.py       # realistic synthetic TB-AMR generator (dev/CI)
│   │   └── download.py        # real CRyPTIC/NCBI acquisition (run on your box)
│   ├── features/
│   │   └── build_features.py  # long tables → model-ready matrices
│   ├── models/
│   │   ├── train.py           # logistic + XGBoost per drug
│   │   └── evaluate.py        # clinical metrics, xgb vs baseline
│   └── interpret/
│       └── explain.py         # SHAP: global importance + per-isolate "why"
├── data/{raw,processed,sample}/
└── reports/                   # metrics.csv + SHAP plots
```

## Quickstart

```bash
# 1. Environment (conda recommended given the GPU/XGBoost setup)
conda create -n tbamr python=3.11 -y && conda activate tbamr
pip install -r requirements.txt

# 2. Run the whole thing on synthetic data (generates it on first run)
python run_pipeline.py

# 3. Inspect a single drug's interpretability
python -m src.interpret.explain --drug Isoniazid
```

XGBoost uses `tree_method="hist"` and runs fine on CPU; set `device="cuda"` in
`config.yaml`'s `xgb_params` to use your GPU on the larger real dataset.

**Switching to real data:** implement `src/data/download.py` (CRyPTIC tables →
the three CSVs), then:

```bash
python src/data/download.py --out data/processed
python run_pipeline.py --data data/processed
```

## Current results (synthetic data, seed 42)

These numbers **validate the pipeline**, not real-world performance — the labels
are synthetic. But the *pattern* mirrors reality:

| Drug | % R | AUC (logistic) | AUC (XGBoost) | Sensitivity | Specificity |
|------|----:|---------------:|--------------:|------------:|------------:|
| Rifampicin   | 25% | 0.882 | 0.873 | 0.76 | 0.94 |
| Isoniazid    | 21% | 0.846 | 0.850 | 0.73 | 0.91 |
| Levofloxacin | 22% | 0.831 | 0.832 | 0.70 | 0.93 |
| Amikacin     | 20% | 0.805 | 0.814 | 0.63 | 0.95 |
| Streptomycin | 21% | 0.806 | 0.793 | 0.63 | 0.92 |
| Ethambutol   | 13% | 0.748 | 0.753 | 0.46 | 0.91 |
| Pyrazinamide | 15% | 0.750 | 0.751 | 0.57 | 0.88 |

Two findings worth noting, both true of real TB-AMR:

1. **Single-marker drugs are easy and essentially linear.** Rifampicin and isoniazid score highest, and SHAP confirms the model relies on the right genes (rpoB, katG) — not lineage background. On these, the logistic baseline is as good as XGBoost.
2. **XGBoost's edge shows up where the biology is non-linear.** Ethambutol carries a built-in epistatic interaction (resistance needs embB M306V *and* G406A together); the tree model captures that AND and edges the linear baseline there, while pyrazinamide's diffuse mechanism stays hard for both.

## Data sources (real)

- **CRyPTIC** — ~15–20k isolates, WGS + MICs across 13 drugs (gold standard): <https://www.crypticproject.org/>
- **WHO mutation catalogue (v2, 2023)** — reference resistance-associated mutations; use to name/validate features and as a rule-based baseline to beat.
- **NCBI Pathogen Detection** — <https://www.ncbi.nlm.nih.gov/pathogens/>
- **BV-BRC (ex-PATRIC)** — AMR phenotypes: <https://www.bv-brc.org/>
- **TB-Profiler** — variant calling from FASTQ/BAM + curated resistance DB: <https://github.com/jodyphelan/TBProfiler>

## Roadmap

- [x] **Phase 0 — Foundation.** Scaffold, source-agnostic pipeline, baseline + XGBoost, clinical metrics, SHAP interpretability, running end-to-end on synthetic data.
- [ ] **Phase 1 — Real data.** Ingest CRyPTIC + WHO catalogue; normalise mutation nomenclature; reproduce per-drug performance; benchmark against the WHO rule-based catalogue.
- [ ] **Phase 2 — A model that stretches me.** A single **multi-task** model predicting all drugs jointly (sharing signal across co-resistant drugs), and a **catalogue-free representation** (k-mer / learned mutation embeddings) so the model can find resistance signal beyond the known catalogue — the genuinely new technique vs. my HOPV work.
- [ ] **Phase 3 — Clinical decision-support layer.** Per-isolate report: predicted regimen, the mutations behind each call, calibrated confidence, and a flag when a call rests on novel/uncatalogued variants.
- [ ] **Phase 4 — Deployment.** A web app: upload a variant profile → per-drug R/S with explanations; containerised, with model versioning and drift monitoring.

## Disclaimer

Research and educational project. Current metrics are on **synthetic data**.
Nothing here is validated for clinical use; real deployment in diagnostics would
require prospective validation and regulatory clearance.
