# TB Antimicrobial-Resistance Predictor — Results

Predicting *Mycobacterium tuberculosis* drug resistance directly from genome
variants, with per-drug models, cross-validated confidence intervals, SHAP
interpretability, MIC regression, population-structure validation, a from-scratch
**regression-based resistance catalogue**, and a head-to-head against the WHO 2023
mutation catalogue. Trained and evaluated on the full **CRyPTIC** reuse compendium
of **12,288 clinical isolates**, with a decision-support frontend on top.

---

## TL;DR

- End-to-end pipeline: raw CRyPTIC VCFs → gene-annotated variant features
  (SNPs + indels + per-gene burden) → per-drug resistance prediction →
  interpretable drivers, plus a Streamlit app that returns a 13-drug profile with
  per-call explanations.
- **First-line resistance is predicted with high, stable accuracy.** 5-fold CV:
  Isoniazid **0.979**, Rifampicin **0.978**, catching **90% of resistant cases at
  ~98% specificity**, with per-fold AUC sd ≤ 0.003.
- **Recommended model restricts features to ~40 resistance genes (+promoters).**
  It matches the genome-wide model on first/second-line using **2,055 features
  instead of ~15,000 (–87%)**, makes even a linear logistic model competitive with
  XGBoost, and — the key finding — **exposes that the genome-wide model's *higher*
  last-line accuracy was lineage confounding, not resistance signal.**
- **Accuracy follows a biologically honest gradient:** first-line ≈ 0.98 →
  second-line ≈ 0.93 → newest last-line drugs 0.65–0.78.
- **Population-structure validation** confirms first-line models learned mechanism,
  not lineage (leave-lineages-out drop ~0.01); Bedaquiline's drop falls from ~0.05
  (genome-wide) to **~0.00** under candidate-gene restriction — its little signal
  now generalises, at a lower, honest AUC.
- **A from-scratch L1-logistic catalogue rediscovers textbook biology** — `rpoB`
  RRDR (rifampicin), `katG` codon 315 (isoniazid), `embB` codon 306 (ethambutol),
  `gyrA` QRDR (fluoroquinolones), `rrs` A1401G **and the *eis* promoter**
  (aminoglycosides) — while its last-line "hits" are dominated by cross-drug
  hitchhikers, giving variant-level evidence for why those drugs are unlearnable.
- **Versus the WHO 2023 catalogue,** ML ties on established drugs and both fail on
  the newest drugs — a result independently reproduced across the 2024–2026
  literature.
- **Regressing MIC beats binary classification for the rarest drugs, and it's
  lineage-verified** — bedaquiline's ranking AUC rises to 0.870 (0.802 under
  leave-lineages-out, still above the binary classifier's 0.777), and for clofazimine
  MIC is far more lineage-robust (0.749 vs 0.614 grouped). The one lever that
  genuinely beats the last-line ceiling — though it's ranking skill, not a usable
  ECOFF operating point.
- **A shippable decision-support app** returns a 13-drug profile with per-call SHAP,
  calibrated probabilities (Brier down up to ~87% on the rare drugs), and
  Walker-style abstention that flags "uncertain" on uncatalogued resistance-gene
  variants instead of assuming susceptible.

---

## Dataset

| Property | Value |
|---|---|
| Source | CRyPTIC reuse release (June 2022), `CRyPTIC_reuse_table_20221019.csv` |
| Isolates | 12,288 clinical *M. tuberculosis* genomes |
| Phenotypes | Binary R/S + MIC per drug, from broth microdilution |
| Label quality filter | HIGH only (agreement across ≥2 MIC assays) |
| Genotype source | Per-sample "masked" VCFs vs H37Rv (NC_000962.3) |
| Variants (SNP-only) | 12.35M rows → 621,357 distinct SNPs |
| Variants (SNP + indel) | 14.45M rows → 806,119 distinct variants |
| Features — genome-wide | 15,000 (floor ≥20 isolates, top-15k) + 43 gene-burden |
| Features — candidate-gene | 2,012 SNP (floor ≥5, in-gene ±200 bp) + 43 gene-burden |
| Drugs modelled | 13 |

Per-drug HIGH-quality isolate counts range from 6,614 (Ethambutol) to 11,926
(Delamanid). No pyrazinamide phenotype exists in this release.

---

## Methodology

**Features.** Each isolate is a binary presence/absence vector over the variants
it carries. Variants in known resistance genes are annotated for readability
(`rpoB@761155_C>T`); others keep a genomic-position token. Controls, tunable
per-run via environment variables, shape the (otherwise 600k–800k-wide, mostly
rare/lineage) feature space:
- **min_count** — prevalence floor (drop variants seen in too few isolates).
- **max_features** — hard ceiling on the most-prevalent variants.
- **feature mode** — per-SNP, per-gene **burden** (all variants in a resistance
  gene pooled into one count), or **both**; plus an INDEL-inclusion toggle.
- **candidate-gene restriction** — keep only variants inside a curated set of ~40
  resistance genes, padded **±200 bp** to capture promoter/upstream mutations
  (e.g. *eis*, *inhA*, *ahpC* promoters). This is the standard approach in the
  field and the primary lever against population-structure confounding.

**Models (per drug, independent).** Gradient-boosted trees (XGBoost),
class-weighted, versus an L2 logistic-regression baseline on a sparse matrix; and,
for the catalogue, an L1-penalised logistic whose coefficients *are* the catalogue.
A shared-trunk multi-task neural network was explored at smaller scale (see
Findings). Sparse matrices and the feature ceiling let the full 12k run complete
on a laptop (RTX 3060 / Ryzen 7).

**Evaluation.** Stratified 5-fold CV with out-of-fold predictions (every isolate
tested once); pooled OOF AUC; bootstrap 95% CI (1,000 resamples); per-fold AUC
mean ± sd; and a sensitivity-targeted operating point (threshold catching ≥90% of
resistant isolates and the specificity it costs), since a missed resistant call is
the clinically dangerous error.

**Interpretability.** Exact TreeSHAP (XGBoost native contributions) for global
per-drug importance and per-isolate "why this call" explanations, corroborated at
scale by the regression catalogue's coefficients.

---

## Headline results — recommended model (candidate-gene + promoters + indels + burden), 5-fold CV

2,055 features (2,012 candidate-gene SNPs + 43 gene-burden), min_count 5.

| Drug | n | %R | AUC (95% CI) | AUC logit | Sens@0.5 | Spec@0.5 | Thr for 90% sens | Spec at that thr |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Isoniazid | 9,519 | 47% | 0.979 (0.976–0.982) | 0.980 | 0.941 | 0.977 | 0.86 | 0.988 |
| Rifampicin | 8,955 | 39% | 0.978 (0.975–0.982) | 0.978 | 0.944 | 0.968 | 0.84 | 0.981 |
| Rifabutin | 10,042 | 32% | 0.977 (0.973–0.980) | 0.975 | 0.924 | 0.969 | 0.72 | 0.982 |
| Ethambutol | 6,614 | 22% | 0.971 (0.966–0.976) | 0.970 | 0.924 | 0.951 | 0.69 | 0.961 |
| Ethionamide | 8,251 | 14% | 0.957 (0.950–0.965) | 0.949 | 0.870 | 0.948 | 0.36 | 0.923 |
| Moxifloxacin | 6,785 | 15% | 0.944 (0.936–0.954) | 0.939 | 0.857 | 0.952 | 0.23 | 0.894 |
| Levofloxacin | 7,774 | 15% | 0.940 (0.931–0.949) | 0.939 | 0.842 | 0.969 | 0.18 | 0.830 |
| Kanamycin | 9,333 | 8% | 0.937 (0.923–0.949) | 0.930 | 0.816 | 0.970 | 0.17 | 0.797 |
| Amikacin | 8,973 | 7% | 0.929 (0.914–0.944) | 0.925 | 0.828 | 0.991 | 0.10 | 0.726 |
| Bedaquiline | 8,536 | 1% | 0.776 (0.698–0.840) | 0.774 | 0.408 | 0.975 | 0.01 | 0.317 |
| Clofazimine | 7,763 | 1% | 0.750 (0.698–0.802) | 0.729 | 0.377 | 0.927 | 0.04 | 0.287 |
| Linezolid | 7,141 | 1% | 0.731 (0.653–0.805) | 0.705 | 0.382 | 0.957 | 0.02 | 0.147 |
| Delamanid | 11,926 | 2% | 0.652 (0.607–0.695) | 0.639 | 0.301 | 0.872 | 0.13 | 0.249 |

On the candidate-gene features the **logistic baseline essentially matches
XGBoost** (e.g. Isoniazid 0.980 vs 0.979) — the curated features are close to
linearly separable, which is itself evidence the signal is real and concentrated.
Per-fold AUC sd is 0.002–0.013 for every drug above 0.9.

---

## Finding 1 — feature representation: indels help the newest drugs, burden alone does not

Genome-wide track, same isolates, AUC (5-fold CV):

| Drug | SNP-only | SNP + burden | SNP + burden + **indels** |
|---|---:|---:|---:|
| Isoniazid | 0.974 | 0.978 | **0.980** |
| Rifampicin | 0.976 | 0.977 | **0.979** |
| Bedaquiline | 0.781 | 0.738 | **0.816** |
| Linezolid | 0.748 | 0.749 | **0.790** |
| Clofazimine | 0.786 | 0.778 | 0.788 |
| Delamanid | 0.588 | 0.619 | 0.617 |

- **Gene-burden alone did not rescue the last-line drugs.** Pooling every variant
  in a resistance gene provably recovers heterogeneous signal (verified on
  synthetic data) but did not help on real data — real genes carry neutral
  polymorphism (burden = signal + noise without amino-acid-consequence filtering),
  and the last-line drugs are limited by resistance *prevalence* (~1%, ≈85–120
  resistant isolates), not representation.
- **Indels genuinely helped the newest drugs** (+0.035 Bedaquiline, +0.042
  Linezolid over SNP-only), matching the biology: loss-of-function frameshift
  indels in *Rv0678* and ribosomal *rrl*/*rplC*, which SNP-only features discard.
  CIs still overlap given low prevalence — improved point estimates, not
  definitive gains.

---

## Finding 2 — candidate-gene restriction: equal accuracy, 87% fewer features, and an honesty correction

Restricting the SNP features to ~40 resistance genes (±200 bp promoters) collapses
the feature space from ~15,000 to **2,012 SNPs**, and produces three results:

**(a) No loss where it matters.** First- and second-line AUCs are unchanged versus
genome-wide (Isoniazid 0.979 vs 0.980, Rifampicin 0.978 vs 0.979, fluoroquinolones
and ethionamide flat) — direct confirmation that the resistance signal lives in the
candidate genes and the other ~13,000 genome-wide features were noise. Foundational
work (Walker et al., 2015) reached the same conclusion: no additional resistance
determinants outside candidate genes.

**(b) A cleaner, more usable model.** The logistic baseline now matches XGBoost, and
several second-line operating points improve at the 90%-sensitivity threshold
(Moxifloxacin specificity 0.894 vs 0.853; Kanamycin 0.797 vs 0.735).

**(c) The last-line AUCs *drop* — and that is the finding.** Bedaquiline
0.816 → 0.776, Linezolid 0.790 → 0.731, Clofazimine 0.788 → 0.750. Cross-referenced
with the population-structure CV below, this shows the genome-wide model's *higher*
last-line numbers were partly built on lineage-background features — strip the
background, and the honest, mechanism-only signal is weaker but generalises. The
genome-wide model was inflating apparent last-line performance via population
structure; candidate-gene restriction gives a lower-but-trustworthy estimate.

This is why candidate-gene is the **recommended** configuration: equal on the drugs
that matter, far more parsimonious and interpretable, and honest on the drugs that
don't work.

---

## Finding 3 — population-structure-aware validation

Genetic clusters were derived from the variant matrix (SVD → k-means, 25 clusters);
models were evaluated with GroupKFold by cluster (test lineages absent from
training) versus random CV. A small drop = signal generalises across genetic
backgrounds (learned resistance, not lineage). Candidate-gene model:

| Drug | AUC (random) | AUC (leave-lineages-out) | drop |
|---|---:|---:|---:|
| Isoniazid | 0.979 | 0.971 | **0.008** |
| Rifampicin | 0.978 | 0.968 | **0.010** |
| Rifabutin | 0.977 | 0.963 | 0.014 |
| Ethambutol | 0.971 | 0.961 | 0.010 |
| Levofloxacin | 0.940 | 0.931 | 0.009 |
| Moxifloxacin | 0.944 | 0.935 | 0.010 |
| Amikacin | 0.929 | 0.928 | **0.001** |
| Ethionamide | 0.957 | 0.932 | 0.026 |
| Kanamycin | 0.937 | 0.899 | 0.038 |
| Bedaquiline | 0.776 | 0.777 | **–0.000** |
| Linezolid | 0.731 | 0.666 | 0.065 |
| Clofazimine | 0.750 | 0.614 | 0.136 |
| Delamanid | 0.652 | 0.609 | 0.043 |

- **First-line: negligible drop (~0.01).** On entirely unseen genetic backgrounds
  the first-line models still work almost perfectly — strong evidence they learned
  the causal mechanism (`rpoB`, `katG`), not lineage. The confounding check a
  reviewer demands, passed cleanly.
- **The Bedaquiline correction.** Genome-wide, Bedaquiline had a leave-lineages-out
  drop of ~0.048 at AUC 0.816; under candidate-gene restriction the drop is
  **~0.000** at AUC 0.776. The dropped 0.04 of AUC *was* the lineage background —
  it did not generalise. What remains is small but real.
- **Last-line remains fragile.** Clofazimine's drop is actually large (0.136);
  at ~1% prevalence (≈80 resistant isolates) these estimates are unstable (wide
  CIs) and partly reflect a few efflux variants that happen to be lineage-linked.
  The consistent message across Findings 2–3: last-line drugs are unusable for two
  compounding reasons — low prevalence *and* population-structure confounding.

---

## Finding 4 — a from-scratch regression catalogue

An L1-penalised logistic regression per drug on the candidate-gene SNPs yields a
sparse, interpretable set of resistance-associated variants (positive coefficient =
resistance-associated; L1 zeroes the uninformative majority) — a *data-driven
catalogue*, following the approach of Hall et al. (*Nature Communications*, 2025).
Each selected variant is cross-referenced against the WHO 2023 catalogue.

| Drug | selected R-variants | WHO-recovered | not in WHO |
|---|---:|---:|---:|
| Rifampicin | 164 | 28 | 136 |
| Isoniazid | 125 | 10 | 115 |
| Ethambutol | 162 | 15 | 147 |
| Moxifloxacin | 183 | 16 | 167 |
| Levofloxacin | 202 | 21 | 181 |
| Kanamycin | 193 | 7 | 186 |
| Amikacin | 141 | 4 | 137 |
| Ethionamide | 260 | 20 | 240 |
| Rifabutin | 166 | 0* | 166 |
| Bedaquiline | 122 | 1 | 121 |
| Clofazimine | 133 | 1 | 132 |
| Linezolid | 86 | 3 | 83 |
| Delamanid | 228 | 1 | 227 |

**It rediscovers textbook biology from nothing but candidate-gene variants.** The
top-coefficient variants per drug are the canonical mechanisms:
- **Rifampicin** → `rpoB` in the 760314–761155 window (the RRDR; 761155 = codon
  450, the textbook S450L).
- **Isoniazid** → `katG@2155168` (codon 315, S315T) + `fabG1`/*inhA*-promoter.
- **Ethambutol** → `embB` around codon 306.
- **Fluoroquinolones** → `gyrA` 7570/7581/7582 (QRDR, codons 90/94) and `gyrB`.
- **Ethionamide** → `inhA`/`fabG1` (its mechanism shared with isoniazid).
- **Aminoglycosides** → `rrs@1473246` (the canonical A1401G) **plus
  `g2715342/2715346/2715369` — the *eis* promoter**, sitting just past the *eis*
  CDS. This is a real, named resistance mechanism (eis promoter → kanamycin) that a
  CDS-only feature set would miss, and it directly validates the ±200 bp promoter
  padding introduced in Finding 2.

**And it exposes, at the variant level, why the last-line drugs fail.** Every drug
selects far more not-in-WHO variants than WHO-graded ones, and the split follows
the prevalence line exactly. For the established drugs the *top* hits are correct
and WHO-graded; the extras are low-coefficient hitchhikers. For the ~1%-prevalence
last-line drugs the list is dominated by **other drugs' genes** — Delamanid's top
"hits" are in `rpoB`, `pncA`, and `rrl` (not its own *fbiA/B/C*, *ddn*, *fgd1*
mechanism); Linezolid's are `pncA`, `ethA`, `embA`. These are co-resistance linkage
and lineage hitchhikers — MDR/XDR-background markers, not causal determinants. This
is the confounding of Finding 3 made legible, and it matches a field caution that
genome-wide models capture markers "more indicative of transmissibility than drug
resistance."

\* Rifabutin shows 0 WHO-recovered only because the WHO 2023 catalogue does not
grade rifabutin as a separate drug; its top hits are the same `rpoB` RRDR variants
as rifampicin (correct cross-resistance) — a cross-reference coverage artifact, not
novelty.

---

## MIC regression — predicting resistance *level*

Regressing log2(MIC) (5-fold CV; essential agreement = within one doubling
dilution):

| Drug | Pearson r | EA ±1 | note |
|---|---:|---:|---|
| Rifabutin | 0.92 | 86% | |
| Isoniazid | 0.89 | 66% | strong correlation, modest exact-dilution accuracy |
| Rifampicin | 0.89 | 62% | RMSE 1.74 dilutions |
| Amikacin | 0.82 | 76% | |
| Ethambutol | 0.78 | 81% | |
| Moxifloxacin | 0.77 | 67% | |
| Levofloxacin | 0.75 | 82% | |
| Kanamycin | 0.75 | 68% | |
| Ethionamide | 0.74 | 78% | |
| Bedaquiline | 0.53 | 69% | |
| Clofazimine | 0.45 | 67% | |
| Linezolid | 0.42 | 81% | high EA trivial (near-uniform low MIC) |
| Delamanid | 0.41 | 63% | |

- **First/second-line MIC level is predictable by correlation** (r 0.74–0.92).
- **A binary AUC of ~0.98 overstates dilution-level precision:** Rifampicin
  r = 0.89 yet essential agreement is only 62%, because RIF MICs are bimodal across
  a ~10-dilution range — direction right, exact dilution not. A useful, humbling
  result about what "solved" means.
- Last-line MIC correlation is weak. XGBoost beat a sparse Ridge baseline on RMSE
  for every drug.

---

## Finding 5 — MIC → binary via ECOFF

Instead of classifying R/S directly, regress log2(MIC) and derive the call by
thresholding *predicted* MIC at the ECOFF, to test whether the quantitative
phenotype carries signal the binary classifier discards. `MIC_AUC` uses predicted
MIC as a resistance score (threshold-free, directly comparable to the direct binary
classifier); the ECOFF is recovered empirically as the true-MIC cut that best
reproduces CRyPTIC's binary phenotype.

| Drug | %R | MIC_AUC | binary_AUC | delta | ECOFF (mg/L) | Sens@ECOFF | Spec@ECOFF |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isoniazid | 47% | 0.978 | 0.979 | −0.001 | 0.14 | 0.937 | 0.973 |
| Rifampicin | 39% | 0.976 | 0.978 | −0.002 | 0.71 | 0.936 | 0.972 |
| Ethambutol | 22% | 0.965 | 0.971 | −0.006 | 4.00 | 0.853 | 0.969 |
| Moxifloxacin | 15% | 0.929 | 0.944 | −0.015 | 1.41 | 0.772 | 0.974 |
| Levofloxacin | 15% | 0.919 | 0.940 | −0.021 | 1.41 | 0.823 | 0.982 |
| Kanamycin | 8% | 0.913 | 0.937 | −0.024 | 5.66 | 0.724 | 0.995 |
| **Bedaquiline** | 1% | **0.870** | 0.776 | **+0.094** | 0.35 | 0.042 | 1.000 |
| **Clofazimine** | 1% | **0.807** | 0.750 | **+0.057** | 0.35 | 0.038 | 0.999 |
| Linezolid | 1% | 0.724 | 0.731 | −0.007 | 1.41 | 0.263 | 1.000 |
| Delamanid | 2% | 0.650 | 0.652 | −0.002 | 0.17 | 0.059 | 1.000 |

- **No benefit where the drugs already work.** Every first/second-line delta is
  slightly negative: when the binary signal is strong, a classifier optimised for
  the R/S decision beats collapsing a regression. The literature's "MIC carries
  extra signal" claim does not hold here.
- **A real ranking gain for the rarest drugs.** Bedaquiline jumps +0.094 (AUC
  0.776 → **0.870**, the best number that drug produces anywhere in this project)
  and Clofazimine +0.057. At ~1% binary prevalence the classifier has ~85 positives
  to learn from, whereas the MIC regressor learns the whole concentration gradient
  across all ~8,500 isolates, recovering ranking signal the binary label discards.
- **The ranking gain survives population-structure control (verified).** Re-running
  the MIC regressor under GroupKFold-by-lineage: bedaquiline's ranking AUC falls
  0.870 → **0.802** (≈0.07 of the naive gain was lineage), but 0.802 still exceeds
  the binary classifier's own leave-lineages-out AUC (**0.777**). Clofazimine is the
  stronger case — MIC 0.749 grouped versus the binary classifier's collapsed 0.614 —
  because the binary model confounds heavily on the rare class while the MIC
  regressor, learning the whole concentration gradient rather than ~80
  lineage-clustered positives, is markedly more lineage-robust. Across all four
  last-line drugs the MIC score beats the binary classifier under leave-lineages-out
  (bedaquiline +0.025, clofazimine +0.135, linezolid +0.048, delamanid +0.024);
  first/second-line MIC drops stay ~0.004–0.015 (mechanism, as expected). So the
  one lever that beats the last-line ceiling is genuine, not a confounding artifact.
- **But it's ranking, not an operating point.** `Sens@ECOFF` for those drugs is ~4%:
  the regressor underpredicts the rare high-MIC tail, so predicted values don't clear
  the breakpoint. Exploiting the gain needs a tuned sub-ECOFF cut, or serving
  predicted MIC as the score for those drugs.

---

## Versus the WHO 2023 mutation catalogue

Rule-based predictor: an isolate is resistant if it carries any variant the
catalogue grades "Assoc w R" / "Assoc w R - Interim", matched by genomic
coordinate. (CRyPTIC encodes some codon changes as phased multi-nucleotide blocks;
these were decomposed to per-position SNPs to match the catalogue's single-position
entries, while the catalogue's own blocks were kept as clean SNPs to avoid injecting
linked phylogenetic markers such as *gyrA* S95T.)

| Drug | Cat Sens | Cat Spec | Cat PPV | ML Sens@0.5 | ML Spec@0.5 |
|---|---:|---:|---:|---:|---:|
| Rifampicin | 0.956 | 0.958 | 0.935 | 0.944 | 0.968 |
| Isoniazid | 0.927 | 0.984 | 0.981 | 0.941 | 0.977 |
| Ethambutol | 0.883 | 0.952 | 0.840 | 0.924 | 0.951 |
| Moxifloxacin | 0.865 | 0.958 | 0.778 | 0.857 | 0.952 |
| Amikacin | 0.841 | 0.992 | 0.881 | 0.828 | 0.991 |
| Levofloxacin | 0.837 | 0.982 | 0.892 | 0.842 | 0.969 |
| Kanamycin | 0.804 | 0.983 | 0.811 | 0.816 | 0.970 |
| Ethionamide | 0.800 | 0.961 | 0.777 | 0.870 | 0.948 |
| Linezolid | 0.316 | 0.999 | 0.857 | 0.382 | 0.957 |
| Bedaquiline | 0.296 | 0.988 | 0.176 | 0.408 | 0.975 |
| Clofazimine | 0.113 | 0.989 | 0.121 | 0.377 | 0.927 |
| Delamanid | 0.108 | 0.998 | 0.513 | 0.301 | 0.872 |

- **Established drugs: a genuine tie.** For the eight first/second-line drugs, ML
  and the catalogue sit within ~0.02–0.06 on sensitivity at matched high
  specificity. A from-scratch ML pipeline reaches the performance of years of
  expert catalogue curation — the central result of this comparison, and unchanged
  by candidate-gene restriction (whose first/second-line AUCs are identical).
- **Newest drugs: both fail, differently.** The catalogue is precise-but-
  insensitive (Linezolid/Delamanid: high PPV, sensitivity 0.11–0.32) or over-calls
  (Bedaquiline/Clofazimine: PPV 0.12–0.18). ML is no better — genotypic prediction
  there is data-limited even for the clinical standard. Independent 2024–2026
  studies report the same: large ensembles decline to attempt bedaquiline,
  delamanid, and clofazimine "because knowledge of the resistance mutations is
  incomplete," and a 2026 benchmark traced bedaquiline's collapse on external data
  to training cohorts drawn from too few sources.

---

## Interpretability

Global SHAP for **Rifampicin** ranks `rpoB@761155` (codon 450, the canonical
RIF-resistance mutation) far above everything else — exactly correct. `katG` (the
*isoniazid* gene) appears second: not a mechanistic effect but **co-resistance
linkage** — RIF resistance almost always co-occurs with INH resistance in MDR-TB,
so the model uses `katG` as an MDR-background marker. The regression catalogue
(Finding 4) corroborates this at scale: the same `rpoB` variants dominate, and the
cross-drug hitchhiking it surfaces is the same correlation, seen across all 13
drugs at once.

---

## Deliverable — decision-support frontend

A Streamlit app (`app.py` + `src/serve/predict.py`) takes an isolate's variants —
pasted tokens or an uploaded VCF — and returns the predicted 13-drug resistance
profile: per-drug R/S call, resistance probability, and an expandable per-call SHAP
explanation. A sidebar toggles the decision threshold between **balanced (0.5,
default)** and **high-sensitivity (catch ~90% of resistance)**, and drugs whose
cross-validated AUC is below 0.85 are flagged **low-confidence**, so the last-line
models are never presented as actionable. Tested end-to-end on susceptible, MDR,
and pre-XDR example isolates.

Two serving-layer safeguards make the displayed output trustworthy:

**Calibrated probabilities.** The models train with class weighting, which ranks
well but inflates probabilities — worst for the rare drugs. A per-drug calibrator
(isotonic where positives allow, Platt otherwise), fit on out-of-fold predictions,
corrects this; both threshold modes then operate on the calibrated scale. The Brier
improvement scales inversely with prevalence, exactly as expected:

| Drug | %R | method | Brier raw → cal | improvement |
|---|---:|---|---|---:|
| Delamanid | 2% | sigmoid | 0.113 → 0.015 | 87% |
| Clofazimine | 1% | sigmoid | 0.064 → 0.013 | 80% |
| Linezolid | 1% | sigmoid | 0.042 → 0.009 | 78% |
| Bedaquiline | 1% | sigmoid | 0.027 → 0.008 | 71% |
| Kanamycin | 8% | isotonic | 0.042 → 0.023 | 44% |
| Amikacin | 7% | isotonic | 0.027 → 0.015 | 43% |
| Moxifloxacin | 15% | isotonic | 0.056 → 0.045 | 20% |
| Rifampicin | 39% | isotonic | 0.036 → 0.035 | 1% |
| Isoniazid | 47% | isotonic | 0.034 → 0.035 | ~0% |

Negligible where the model was already honest (the balanced first-line drugs),
transformative where class weighting had inflated it (the 1%-prevalence drugs) —
the same inflation that once made a susceptible isolate read as 53% linezolid-resistant.

**Abstention on uncatalogued variants.** Each drug has a mechanism-gene map; when an
isolate carries a variant inside one of a drug's resistance genes that the model
never saw in training, that drug is flagged **Uncertain** rather than called
susceptible — the Walker-style safe default, since a variant of unknown effect in a
resistance gene cannot be assumed benign. Confident resistant calls (driven by known
variants) and drugs whose genes are untouched are unaffected.

---

## Limitations

- **Single data source** (CRyPTIC); no external-cohort validation. Structure-aware
  CV mitigates but does not replace this — and the field agrees it is the key gap:
  even large ensemble studies report being unable to find a suitable external
  dataset for second-line drugs.
- **Probabilities are calibrated** per drug (isotonic/Platt on out-of-fold
  predictions); the raw class-weighted scores are well-ranked but inflated, so the
  served probabilities and both threshold modes use the calibrated scale.
- **The MIC regressor's last-line gain is population-structure-verified** — under
  leave-lineages-out it still beats the binary classifier for all four last-line
  drugs (bedaquiline 0.802 vs 0.777, clofazimine 0.749 vs 0.614), though bedaquiline's
  headline falls from the naive 0.870 to 0.802.
- **Last-line drugs are not usable** from genotype here — prevalence (~1%) *and*
  population-structure confounding, now demonstrated three ways (ablation,
  structure-aware CV, and the regression catalogue's hitchhikers).
- **Retrospective, imperfect labels.** Predicts a binary/MIC phenotype from broth
  microdilution; DST for the newest drugs is itself noisy. Not a validated device.

---

## Reproducibility

```bash
# 1. Phenotypes + isolate index
python -m src.data.download --reuse-table data/processed/CRyPTIC_reuse_table_20221019.csv

# 2. Download + parse all VCFs (resumable). TBAMR_INDELS=1 builds the indel set.
python -m src.data.vcf_fetch --all --out data/vcf \
  --ftp-base "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reproducibility/"
TBAMR_INDELS=1 python -m src.data.vcf_fetch --all --out data/vcf_indel --cache data/vcf/cache \
  --ftp-base "https://ftp.ebi.ac.uk/pub/databases/cryptic/release_june2022/reproducibility/"

# 3a. Genome-wide track (feature-representation ablation, Finding 1)
export TBAMR_MIN_COUNT=20 TBAMR_MAX_FEATURES=15000 TBAMR_FEATURES=both
python -m src.models.evaluate_cv --data data/vcf_indel --folds 5 --target-sens 0.90

# 3b. Recommended model — candidate-gene + promoters (Findings 2–3)
export TBAMR_MIN_COUNT=5 TBAMR_MAX_FEATURES=15000 TBAMR_FEATURES=both \
       TBAMR_CANDIDATE_ONLY=1 TBAMR_GENE_PAD=200
python run_pipeline.py           --data data/vcf_indel
python -m src.models.evaluate_cv --data data/vcf_indel --folds 5 --target-sens 0.90
python -m src.models.lineage_cv  --data data/vcf_indel --clusters 25 --folds 5

# 4. MIC regression, MIC->binary, WHO baseline, regression catalogue
python -m src.models.mic_regress          --data data/vcf_indel --reuse-table data/processed/CRyPTIC_reuse_table_20221019.csv
python -m src.models.mic_to_binary        --data data/vcf_indel --reuse-table data/processed/CRyPTIC_reuse_table_20221019.csv --ml-metrics reports/cv_metrics.csv
python -m src.models.mic_lineage_cv       --data data/vcf_indel --reuse-table data/processed/CRyPTIC_reuse_table_20221019.csv --clusters 25 --folds 5
python -m src.models.who_baseline         --data data/vcf_indel --catalogue data/processed/WHO-UCN-TB-2023.7-eng.xlsx --ml-metrics reports/cv_metrics.csv
python -m src.models.regression_catalogue --data data/vcf_indel --catalogue data/processed/WHO-UCN-TB-2023.7-eng.xlsx --top-k 12

# 5. Calibrate the served model (writes models/calibrators.joblib), then serve
python -m src.models.calibrate            --data data/vcf_indel
streamlit run app.py
```

Environment: Python 3.10, scikit-learn, XGBoost, SHAP, pandas, NumPy, openpyxl,
Streamlit.

---

## Future work

1. **External validation on a non-CRyPTIC cohort** — the single highest-value
   addition and the field's acknowledged gap. Acquire a phenotyped independent
   cohort (e.g. SRA isolates from a different geography), run variant calling from
   raw reads, and re-predict without retraining. This is what would let the
   generalisation claims stand unqualified.
2. **Convert the MIC ranking gain into usable calls for the rarest drugs** — the MIC
   regressor's advantage on bedaquiline/clofazimine is now population-structure-verified
   (Finding 5), but it underpredicts the high-MIC tail so the ECOFF is too
   conservative. A tuned sub-ECOFF threshold, or serving predicted MIC as the score
   for those drugs in the app, would turn the verified ranking gain into an
   operating point.
3. **Per-drug mechanism-gene-restricted catalogue** — map each drug to only its own
   mechanism genes to remove the cross-drug hitchhiking in Finding 4 and produce a
   cleaner per-drug variant list.

*Completed since the first draft:* candidate-gene restriction with promoter windows
(now the recommended model), a from-scratch regression catalogue, MIC → binary via
ECOFF (Finding 5), per-drug probability calibration, and abstention on uncatalogued
variants (both live in the frontend).

*Deprioritised:* amino-acid-consequence-filtered burden — Findings 1–3 show the
last-line ceiling is prevalence and confounding, not representation, so this has
little headroom despite being the intuitive next step.
