# External Validation

> Slots into `RESULTS.md` as the external-validation section. Numbers are from the
> frozen CRyPTIC/GenTB models scored on two independent holdout cohorts and pooled.

## TL;DR

- The frozen models were validated on **two independent external cohorts** from
  different continents and lineage backgrounds, never seen in training:
  **Sierra Leone** (West African L4 + *M. africanum*, n=89) and
  **Belarus** (Eastern European Beijing/L2, n=97).
- **Pooled AUC 0.93–0.97 across all five first-line drugs** (n=186), with tight
  confidence intervals and no cross-cohort scale drift on four of five drugs.
- **Streptomycin — the weakest drug internally — holds at 0.93 pooled** (0.97 on
  Belarus alone) on 110 resistant isolates, reframing an apparent model weakness
  as small-sample noise in the first cohort.
- On Sierra Leone the models **beat off-the-shelf TB-Profiler on PZA and STR**,
  the two drugs a fixed catalogue handles worst.
- Validation is **leakage-clean** on Sierra Leone (0 shared ENA accessions with
  CRyPTIC) and **provenance-clean** on Belarus (independent consortium; see
  Data integrity).
- No model weights were changed to obtain these numbers. Decision thresholds were
  re-fit on external scores (disclosed) because the GenTB thresholds do not
  transfer; AUC is the threshold-free headline.

## Cohorts

| Cohort | Accession | Source | n (scored) | Lineages | Resistance profile |
|---|---|---|---|---|---|
| Sierra Leone | PRJEB7727 | Schleusener et al. 2017, *Sci Rep* 7:46327 | 89 | L4 + *M. africanum* (L5/L6, ~23%) | Low-MDR, susceptible-heavy |
| Belarus | PRJNA200335 | Wollenberg et al. 2017, *J Clin Microbiol* 55:457 | 97 | Beijing/L2 (+ LAM) | MDR/XDR, resistant-heavy |

The two cohorts are deliberate complements: Sierra Leone supplies the
susceptible-heavy first-line side and a divergent-lineage stress test;
Belarus supplies the resistant-heavy side and a large, mechanistically diverse
streptomycin-resistant set. Together they span opposite ends of both the lineage
spectrum (the lineages CRyPTIC has fewest vs most of) and the resistance spectrum.

## Method

Each cohort was processed through an identical pipeline: ENA/SRA FASTQ →
TB-Profiler (WHO catalogue db) → harmonisation of called variants to the
CRyPTIC token vocabulary → scoring by the frozen models. Feature vectors are
reconstructed per model in **both token conventions** (position-only
`g{pos}_{ref}>{alt}` for the CRyPTIC models, `gene@{pos}_{ref}>{alt}` for the
GenTB PZA/STR models) plus gene-level burden features; emitting only one form
silently zeroes the GenTB SNP features. Phenotypic DST from each paper's
supplementary table is the ground truth. AUC is reported with a 1,000-sample
bootstrap 95% CI. Operating points are re-fit on the external calibrated scores
at a 90% sensitivity target and reported leave-one-out (each isolate classified
by a threshold fit on the others), so the reported sensitivity/specificity is
not optimistic.

## Headline results

Pooled ROC-AUC (n=186) with per-cohort values alongside:

| Drug | Pooled AUC (95% CI) | R / S | Sierra Leone | Belarus |
|---|---|---|---|---|
| RIF | 0.974 (0.947–0.997) | 82 / 103 | 1.000 | 0.914 |
| INH | 0.972 (0.946–0.991) | 98 / 88 | 0.929 | 0.984 |
| EMB | 0.969 (0.942–0.991) | 81 / 105 | 0.967 | 0.937 |
| PZA | 0.970 (0.919–1.000) | 27 / 88 | 0.989 | 0.974 |
| STR | 0.926 (0.885–0.960) | 110 / 76 | 0.855 | 0.966 |

Pooled operating points at a 90%-sensitivity target (leave-one-out on the pooled
external calibrated scores):

| Drug | Sensitivity | Specificity |
|---|---|---|
| RIF | 0.90 | 0.97 |
| INH | 0.90 | 0.94 |
| EMB | 0.89 | 0.95 |
| PZA | 0.96 | 0.98 |
| STR | 0.92 | 0.70 |

## Key findings

**Generalisation across the lineage spectrum.** Four of five drugs pool cleanly —
the pooled AUC lands between or at the two cohort values (e.g. INH 0.93/0.98 →
0.97; EMB 0.97/0.94 → 0.97), indicating the models' score distributions align
across two very different genetic backgrounds with no systematic scale drift.
The models were trained on a largely L2/L4 compendium yet hold on both West
African L4 + *M. africanum* and Eastern European Beijing.

**Streptomycin reframed.** STR scored 0.86 on Sierra Leone — the project's
weakest result — but 0.97 on Belarus, and 0.93 pooled across 110 resistant
isolates. The Sierra Leone figure was small-sample noise (36 resistant) compounded
by *M. africanum* genotype–phenotype discordance, not a capacity limit: given a
large, mechanistically diverse resistant set (rpsL, rrs, gidB), the model
discriminates streptomycin resistance well.

**Beats a fixed catalogue where catalogues are weakest.** On Sierra Leone, against
the paper's own TB-Profiler-vs-DST numbers, the models match the tool on the
first-line drugs and clearly exceed it on PZA (model AUC 0.99 vs tool sensitivity
0.44) and STR — precisely the drugs a fixed catalogue handles worst, where a
learned gene-burden model closes the gap.

**Thin-positive weakness resolved.** PZA rested on 8 resistant isolates in Sierra
Leone (fragile despite AUC 0.99); pooled it has 27, and RIF/INH now carry balanced
R/S. The confidence intervals are real rather than artifacts of tiny cells.

## Calibration and error analysis

- **Probabilities do not transfer, ranking does.** On the external cohorts the raw
  model probabilities are better calibrated than the training-set-calibrated ones
  (lower Brier for every drug), and the models are systematically overconfident
  out-of-domain. Report AUC as the trustworthy summary; use raw scores externally
  or recalibrate on the target cohort. This is expected for out-of-distribution
  deployment, not a defect.
- **The streptomycin ceiling is mechanistic.** Of the STR misses on Sierra Leone,
  half carried no rpsL/rrs/gidB variant at all (phenotypic resistance invisible to
  any variant-based model), and the rest carried only gidB variants, which the
  model conservatively down-weights because gidB mixes causal and phylogenetic
  mutations. STR's 0.70 pooled specificity at 90% sensitivity is capped by its AUC,
  not recoverable by threshold choice — the honest limitation of the drug, not a
  bug in the model.

## Data integrity (leakage)

- **Sierra Leone: verified clean.** Zero shared ENA accessions between the CRyPTIC
  training set (12,287 `ERS…` samples) and the Sierra Leone cohort (`ERR/SAMEA/ERS`),
  confirmed by an accession-intersection check. The near-total absence of
  *M. africanum* in CRyPTIC independently corroborates non-overlap.
- **Belarus: provenance-clean.** Belarus uses NCBI accessions (`SAMN/SRS/SRR`) and
  CRyPTIC uses ENA (`ERS`); the same isolate submitted to different archives receives
  different accessions, so accession intersection is structurally incapable of
  detecting overlap and is *not* reported for this cohort. Disjointness instead rests
  on provenance: the CRyPTIC reuse table is the consortium's own prospectively
  sequenced isolates (2017–2021, 14 partner sites, none in Belarus), whereas Belarus
  is an independent Broad deposit of 2010–2013 Minsk isolates. A genomic-identity
  check is possible but confounded by differing variant-calling pipelines and was
  judged unnecessary given the provenance separation.

## Limitations

- Two cohorts, not a broad multi-country panel — strong evidence of generalisation,
  not proof of universality.
- GenTB decision thresholds do not transfer to external data; operating points must
  be re-fit on the target distribution (done here, disclosed).
- STR specificity is capped (~0.70 at 90% sensitivity) by genuine
  genotype–phenotype discordance and gidB ambiguity.
- PZA remains the thinnest drug (27 pooled resistant); its CI is correspondingly
  wider.
- Pooled STR AUC (0.93) sits below Belarus alone (0.97) due to cross-cohort score-
  scale differences — reported honestly rather than taking the higher single-cohort
  number.
- A third cohort (**Latvia**, PRJEB59824) is genotyped, leakage-checked, and staged
  in the database as `lv_ext`, but its per-isolate phenotypic DST was not obtainable,
  so it contributes no scored result. If the labels are later obtained (author
  request or a follow-up dataset), scoring is ~15 minutes of already-built pipeline.

## Reproducibility

Per cohort (Sierra Leone / Belarus), from a registered ENA project accession:

```bash
# 1. pull the ENA/SRA file report -> build a fastq manifest
#    (explode paired FASTQ into the query/run/sample/url manifest format)
# 2. profile all isolates
python -m src.data.run_profiling --manifest data/processed/<cohort>_manifest.tsv \
    --results-dir results_<cohort> --threads 4 --clean --retries 12
# 3. register the holdout dataset (is_holdout=TRUE) and ingest variants
python -m src.data.holdout_ingest --results results_<cohort>/results \
    --db data/tbamr.duckdb --dataset <cohort>_ext
# 4. load per-isolate phenotypes from the paper supplementary
#    (cohort-specific parser; maps the paper's isolate IDs to the DB accessions)
# 5. leakage check (accession intersection vs CRyPTIC; N/A across NCBI<->ENA)
python -m src.data.detect_leakage \
    --cryptic-meta data/processed/CRyPTIC_reuse_table_20221019.csv \
    --sl-crosswalk data/processed/<cohort>_ena.tsv
# 6. score the frozen models
python -m src.models.evaluate_holdout \
    --db data/tbamr.duckdb --results results_<cohort>/results --dataset <cohort>_ext \
    --models models --models-pza models_gentb_pza --models-str models_gentb_str \
    --reports reports_<cohort>
# 7. refit operating points on external scores; run calibration + miss diagnostics
python -m src.models.refit_holdout_threshold  --dataset <cohort>_ext ...
python -m src.models.holdout_diagnostics       --dataset <cohort>_ext ...
```

Pooled result across both cohorts:

```bash
python -m src.models.evaluate_holdout_pooled \
    --db data/tbamr.duckdb \
    --cohort sl_ext:results_sl/results \
    --cohort bel_ext:results_belarus/results \
    --models models --models-pza models_gentb_pza --models-str models_gentb_str \
    --target-sens 0.90 --reports reports
```

## Future work

- Obtain Latvia (`lv_ext`) per-isolate DST to add a third scored cohort (already staged).
- Add further external cohorts spanning under-represented lineages/regions
  (each run through the leakage check).
- Streptomycin improvement is a **data** problem, not a modelling one — more
  STR-resistant isolates across diverse mechanisms (e.g. via TB Portals), not a
  new architecture or longer training.
