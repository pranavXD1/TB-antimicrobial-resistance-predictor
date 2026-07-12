# TB Drug-Resistance Predictor (`tbamr`)

Genome-based antimicrobial-resistance (AMR) prediction for *Mycobacterium tuberculosis*. Given an isolate's genomic variants (pasted or as a VCF), the app estimates resistance across **13 antitubercular drugs**, with calibrated probabilities, per-prediction explanations, and results from **external validation on two independent cohorts**.

**🔗 Live app:** https://tb-antimicrobial-resistance-predictor.onrender.com
*(Free tier — the first request after a period of inactivity takes ~30–50 s to wake the server.)*

> ⚠️ **Research tool — not a clinical device.** Predictions are not validated for patient care. Confirm any result with phenotypic drug-susceptibility testing (DST).

---

## What it does

- **Predicts resistance for 13 drugs** — first-line (isoniazid, rifampicin, rifabutin, ethambutol), second-line (levofloxacin, moxifloxacin, ethionamide, kanamycin, amikacin), and newer/last-line agents (bedaquiline, clofazimine, linezolid, delamanid).
- **Two decision modes** from a single scored pass: **Balanced** (threshold 0.50, maximises accuracy) and **High-sensitivity** (threshold 0.10, catches more resistance at the cost of more false positives).
- **Explains every call** — per-prediction SHAP-style attributions show which variants drove the result, whether each was present or absent, and its signed contribution.
- **Flags uncertainty** — abstains when an isolate carries an uncatalogued variant in a resistance gene rather than guessing.
- **Shows its own validation** — an in-app dashboard reports pooled AUCs with bootstrap confidence intervals and per-cohort ROC curves.

---

## Results

External validation on two cohorts never seen in training (pooled **n = 186**):

| Drug          | Pooled AUC (95% CI) | Sierra Leone | Belarus | Sens / Spec @90% |
|---------------|:-------------------:|:------------:|:-------:|:----------------:|
| Isoniazid     | 0.97 (0.95–0.99)    | 0.93         | 0.98    | 0.90 / 0.94      |
| Rifampicin    | 0.97 (0.95–1.00)    | 1.00         | 0.91    | 0.90 / 0.97      |
| Ethambutol    | 0.97 (0.94–0.99)    | 0.97         | 0.94    | 0.89 / 0.95      |
| Pyrazinamide  | 0.97 (0.92–1.00)    | 0.99         | 0.97    | 0.96 / 0.98      |
| Streptomycin  | 0.93 (0.89–0.96)    | 0.85         | 0.97    | 0.92 / 0.70      |

Discrimination transfers well across two genetically distant settings — a West-African cohort (lineage 4 + *M. africanum*) and an Eastern-European high-MDR cohort (Beijing / lineage 2).

---

## How it works

### Training data
- **CRyPTIC** — 12,288 clinical *M. tuberculosis* isolates with matched whole-genome sequencing and quantitative (MIC-based) phenotypes across 13 drugs; a deliberately resistance-enriched global panel.
- **GenTB** — per-drug candidate-gene matrices, used to supplement pyrazinamide and streptomycin where CRyPTIC signal is sparse.
- Phenotypes are binarised to resistant / susceptible per drug. Wild-type isolates are retained as all-zero feature rows so each model learns the true background rate.

### Feature representation
- Variants are encoded as position-anchored tokens in two coordinate forms — gene-relative (`gene@pos_ref>alt`) and genome-position against the H37Rv reference (`NC_000962.3`) — so calls from different variant callers harmonise onto the same feature space.
- The space is restricted to **candidate-gene variants** at established resistance loci (`rpoB`, `katG`, `inhA`/`fabG1`, `embA`/`embB`, `pncA`, `gyrA`/`gyrB`, `rrs`/`rpsL`/`gid`, `eis`, `ethA`, `Rv0678`/`atpE`/`pepQ`, `rrl`/`rplC`, `ddn`/`fbiC`…), plus per-gene mutation-burden features. This keeps the models interpretable and portable, at the deliberate cost of ignoring mechanisms outside the panel.

### Models
- One **XGBoost** gradient-boosted classifier per drug, trained independently.
- Raw scores are mapped through **per-drug probability calibrators** (isotonic / Platt) so a reported 0.9 means roughly what it says on held-out data.
- Per-drug **90%-sensitivity operating points** are derived by cross-validation.
- Per-prediction attributions expose the variant-level drivers behind each call.

### External-validation pipeline
1. Pull raw sequencing reads for an independent cohort from the ENA.
2. Call variants with **TB-Profiler** against H37Rv.
3. Harmonise called variants into the CRyPTIC candidate-gene token space.
4. Score with the **frozen** model weights — no retraining, ever.
5. Re-derive decision thresholds on the external calibrated scores by leave-one-out CV (training thresholds don't transfer for the GenTB drugs).
6. Leakage check: verify zero overlap by ENA accession where archives are comparable, and by dataset provenance where they are not.

### Validation cohorts
- **Sierra Leone** — `PRJEB7727` (Schleusener et al., 2017), ~89 usable isolates, West-African lineage 4 with a substantial *M. africanum* (L5/6) fraction.
- **Belarus** — `PRJNA200335` (Wollenberg et al., 2017), ~97 isolates, Beijing / lineage 2, a very high-MDR setting; an independent Broad deposit disjoint from CRyPTIC's prospective sequencing.

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
# open http://localhost:7860
```

---

## Interpreting the output

- **AUC** — threshold-free discrimination: the probability the model ranks a random resistant isolate above a random susceptible one. This is the headline metric because it transfers across cohorts.
- **Sensitivity / specificity @90%** — the operating point that catches 90% of true resistance, and the specificity you pay for it.
- **95% CIs** — bootstrap over pooled isolates (500 resamples).
- **Calibrated probability** vs. **decision threshold** — the probability is the model's confidence; the threshold (0.50 or 0.10) decides the call. Switching modes re-reads the same probabilities against a different threshold.

---

## Limitations & honest caveats

- **Not clinically validated.** This is a portfolio / research project, not a diagnostic.
- **Calibration drifts out-of-distribution.** Discrimination (AUC) transfers, but absolute probabilities tend to be overconfident on new settings — recalibrate per setting before any threshold-based decision.
- **Candidate-gene scope.** The models only see variants at known resistance loci; novel or off-panel mechanisms are invisible to them.
- **Two cohorts ≠ universal.** Strong performance on Sierra Leone and Belarus is encouraging but not proof of global generalisation.
- **Lineage coverage.** *M. africanum*'s lineage-specific SNPs reduce token overlap with the L2/L4-dominated training set, which can lower performance for those isolates.
- **Streptomycin specificity** is capped by a wider genotype–phenotype gap for that drug.

---

## Data & reproducibility

All inputs are public: CRyPTIC (ENA/EBI), GenTB, and the validation cohorts `PRJEB7727` and `PRJNA200335`. Code and small derived artifacts live in this repository; the trained models are included for inference. Every external result is regenerable end-to-end from the pipeline described above.

## Acknowledgements

Built on open data from the **CRyPTIC Consortium**, **GenTB**, and the **European Nucleotide Archive**, with variant calling by **TB-Profiler**. Thanks to the authors of the Sierra Leone (Schleusener et al., 2017) and Belarus (Wollenberg et al., 2017) cohort studies for making their sequencing data public.

## License

MIT — see [`LICENSE`](LICENSE).
