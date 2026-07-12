# TB Drug-Resistance Predictor (`tbamr`)

Genome-based antimicrobial-resistance (AMR) prediction for *Mycobacterium tuberculosis*. Given an isolate's genomic variants (pasted or as a VCF), the app estimates resistance across **13 antitubercular drugs**, with calibrated probabilities, a per-prediction explanation, and results from **external validation on two independent cohorts**.

**🔗 Live app:** https://tb-antimicrobial-resistance-predictor.onrender.com
*(Free tier — the first request after a period of inactivity takes ~30–50 s to wake the server.)*

> ⚠️ **Research tool — not a clinical device.** Predictions are not validated for patient care. Confirm any result with phenotypic drug-susceptibility testing (DST).

---

## Scale

The pipeline processes **~15 GB of genomic data**: **14.45 million variant calls** across **12,288 clinical isolates**, reduced to **806,119 distinct variants**, plus raw per-cohort sequencing pulled from the ENA. It's streamed through a **DuckDB** feature store (native CSV streaming, so the 14 M-row variant table never has to be loaded into RAM at once) before modelling.

---

## What it does

- **Predicts resistance for 13 drugs** — first-line (isoniazid, rifampicin, rifabutin, ethambutol), second-line (levofloxacin, moxifloxacin, ethionamide, kanamycin, amikacin), and newer/last-line agents (bedaquiline, clofazimine, linezolid, delamanid).
- **Two decision modes** from a single scored pass: **Balanced** (threshold 0.50, maximises accuracy) and **High-sensitivity** (threshold 0.10, catches more resistance at the cost of more false positives).
- **Explains every call** — per-prediction SHAP attributions show which variants drove the result, whether each was present or absent, and its signed contribution.
- **Flags uncertainty** — abstains when an isolate carries an uncatalogued variant in a resistance gene rather than guessing.
- **Shows its own validation** — an in-app dashboard reports pooled AUCs with bootstrap confidence intervals and per-cohort ROC curves.

---

## Results

### External validation (headline)

Two cohorts never seen in training, pooled **n = 186** — a West-African cohort (lineage 4 + *M. africanum*) and an Eastern-European high-MDR cohort (Beijing / lineage 2):

| Drug | Pooled AUC (95% CI) | R / S | Sierra Leone | Belarus | Sens / Spec @90% |
|------|:---:|:---:|:---:|:---:|:---:|
| Isoniazid | 0.97 (0.94–0.99) | 98 / 88 | 0.92 | 0.97 | 0.92 / 0.94 |
| Rifampicin | 0.98 (0.95–1.00) | 82 / 103 | 1.00 | 0.92 | 0.90 / 0.97 |
| Ethambutol | 0.97 (0.94–0.99) | 81 / 105 | 0.97 | 0.93 | 0.90 / 0.95 |
| Pyrazinamide | 0.97 (0.92–1.00) | 27 / 88 | 0.99 | 0.97 | 0.96 / 0.98 |
| Streptomycin | 0.92 (0.88–0.95) | 110 / 76 | 0.85 | 0.96 | 0.92 / 0.70 |

*AUC = area under the ROC curve (threshold-free ranking quality). CIs are 2,000-sample bootstrap over pooled isolates. Discrimination transfers across two genetically distant settings — the part that actually matters.*

### Internal cross-validation (all 13 drugs, 5-fold)

Accuracy follows a **biologically honest gradient** — the model is strong exactly where the biology is well-understood, and it doesn't pretend otherwise:

- **First-line ≈ 0.98** — isoniazid 0.979, rifampicin 0.978, rifabutin 0.977, ethambutol 0.971
- **Second-line ≈ 0.93–0.96** — ethionamide 0.957, moxifloxacin 0.944, levofloxacin 0.940, kanamycin 0.937, amikacin 0.929
- **Newest last-line 0.73–0.78** — bedaquiline 0.776, clofazimine 0.750, linezolid 0.731

The last-line numbers are low **on purpose**: those drugs have ~1% resistance in the data (too few resistant isolates to learn from), and the honest model refuses to inflate them. A leave-lineages-out test confirms the models learned resistance biology, not genetic background — first-line AUC drops only ~0.008–0.010 when whole lineages are held out.

---

## How it works

### Training data
- **CRyPTIC** — 12,288 clinical *M. tuberculosis* isolates (reuse release, June 2022) with matched whole-genome sequencing and quantitative, MIC-based phenotypes across 13 drugs; HIGH-quality labels only (agreement across ≥2 MIC assays).
- **GenTB** — per-drug candidate-gene matrices, used to supplement pyrazinamide and streptomycin.
- Phenotypes binarised to resistant / susceptible per drug; wild-type isolates retained as all-zero feature rows so each model learns the true background rate.

### Feature representation
- Each isolate is a **binary presence/absence vector** over the variants it carries, called against the H37Rv reference (`NC_000962.3`). Variants in known resistance genes are annotated for readability (`rpoB@761155_C>T`); others keep a genomic-position token, so calls from different callers harmonise onto one feature space.
- Two configurations were compared: a **genome-wide** set (~15,000 top variants + 43 gene-burden features) and a **candidate-gene** set (2,012 SNPs within ~40 resistance genes ±200 bp promoters + 43 gene-burden = 2,055 features). The candidate-gene model **matches the genome-wide one on first/second-line drugs with 87% fewer features** — and is more honest on the last-line drugs, because it can't lean on lineage-background signal.

### Models
- One **XGBoost** gradient-boosted classifier per drug, trained independently.
- Raw scores mapped through **per-drug probability calibrators** (isotonic / Platt) so a reported 0.9 means roughly what it says on held-out data.
- Per-drug **90%-sensitivity operating points** derived by cross-validation (a missed resistant call is the costly error, so sensitivity is prioritised).
- Per-prediction attributions expose the variant-level drivers behind each call.

### External-validation pipeline
1. Pull raw sequencing reads for an independent cohort from the ENA.
2. Call variants with **TB-Profiler** against H37Rv.
3. Harmonise called variants into the CRyPTIC token space.
4. Score with the **frozen** model weights — no retraining, ever.
5. Re-derive decision thresholds on the external calibrated scores by leave-one-out CV.
6. Leakage check: verify zero overlap by ENA accession where archives are comparable, and by dataset provenance where they are not.

### Validation cohorts
- **Sierra Leone** — `PRJEB7727` (Schleusener et al., 2017), ~89 isolates, West-African lineage 4 with a substantial *M. africanum* (L5/6) fraction.
- **Belarus** — `PRJNA200335` (Wollenberg et al., 2017), ~97 isolates, Beijing / lineage 2, a very high-MDR setting.

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Modelling | XGBoost, scikit-learn (calibration), SHAP |
| Data | CRyPTIC, GenTB, ENA; TB-Profiler for external variant calling; DuckDB feature store |
| Backend | FastAPI + Uvicorn (Python 3.11) |
| Frontend | Vanilla JS / CSS single-page app (no framework) |
| Packaging / deploy | Docker → Render (CPU, free tier) |

### API

| Method | Route | Purpose |
|--------|-------|---------|
| `GET`  | `/`             | Predictor + validation dashboard |
| `POST` | `/predict`      | `{text, mode, threshold}` → per-drug results |
| `POST` | `/predict_vcf`  | Multipart VCF upload → per-drug results |
| `GET`  | `/dashboard`    | Pooled external-validation metrics (AUC, CIs, ROC points) |
| `GET`  | `/health`       | Liveness check |

---

## Run locally

```bash
git clone https://github.com/pranavXD1/TB-antimicrobial-resistance-predictor.git
cd TB-antimicrobial-resistance-predictor

pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```

Or with Docker:

```bash
docker build -t tbamr .
docker run -p 7860:7860 tbamr
```

---

## Interpreting the output

- **AUC** — threshold-free discrimination: the probability the model ranks a random resistant isolate above a random susceptible one. Headline metric because it transfers across cohorts.
- **Sensitivity / specificity @90%** — the operating point that catches 90% of true resistance, and the specificity you pay for it.
- **Calibrated probability** vs. **decision threshold** — the probability is the model's confidence; the threshold (0.50 or 0.10) decides the call. Switching modes re-reads the same probabilities against a different threshold.

---

## Limitations & honest caveats

- **Not clinically validated.** A research / portfolio project, not a diagnostic.
- **Calibration drifts out-of-distribution.** Discrimination (AUC) transfers, but absolute probabilities tend to be overconfident on new settings — recalibrate per setting before any threshold-based decision.
- **Last-line drugs are weak by design** (bedaquiline / clofazimine / linezolid, ~0.73–0.78) — too few resistant isolates exist to learn from, and the model doesn't hide it.
- **Two cohorts ≠ universal.** Strong performance on Sierra Leone and Belarus is encouraging, not proof of global generalisation.
- **Candidate-gene scope.** The models only see variants at known resistance loci; novel or off-panel mechanisms are invisible to them.
- **Streptomycin specificity** is capped by a wider genotype–phenotype gap for that drug.

---

## Data & reproducibility

All inputs are public: CRyPTIC (ENA/EBI), GenTB, and the validation cohorts `PRJEB7727` and `PRJNA200335`. Code and small derived artifacts live in this repository; the trained models are included for inference. Every external result is regenerable end-to-end from the pipeline above.

## Acknowledgements

Built on open data from the **CRyPTIC Consortium**, **GenTB**, and the **European Nucleotide Archive**, with variant calling by **TB-Profiler**. Thanks to the authors of the Sierra Leone (Schleusener et al., 2017) and Belarus (Wollenberg et al., 2017) cohort studies for making their sequencing data public.

## License

MIT — see [`LICENSE`](LICENSE).
