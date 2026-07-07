"""
Biologically-realistic synthetic generator for M. tuberculosis AMR data.

WHY THIS EXISTS
---------------
The real datasets (CRyPTIC, NCBI, BV-BRC) live behind FTP/figshare and can't be
pulled in every sandbox. This module fabricates data with the *same schema and
statistical shape* as the real thing so the entire pipeline runs end-to-end for
development, testing, and CI. On a machine with network access you replace this
with `src/data/download.py`, which emits the identical CSVs, and nothing
downstream changes.

WHAT MAKES IT REALISTIC
-----------------------
* Real resistance genes & canonical mutations (rpoB->RIF, katG->INH, ...).
* Partial penetrance: a mutation raises P(resistant) but doesn't guarantee it.
* "Unknown mechanism" resistance + phenotyping error, so the problem is NOT
  perfectly separable (mirrors real TB: rifampicin is near-solved, pyrazinamide
  is hard).
* Population structure (lineages carry characteristic neutral background
  mutations) -> a real-world confounder the model must see past.
* Co-resistance / MDR clustering: acquiring one resistance mutation makes others
  more likely, reproducing MDR-TB patterns.
* Incomplete drug-susceptibility testing: not every isolate is tested for every
  drug, so phenotype tables have gaps -- exactly like clinical data.

OUTPUT (long-format CSVs, mirroring real processed data)
--------------------------------------------------------
* variants.csv     : (isolate_id, mutation)        -- one row per *present* mutation
* phenotypes.csv   : (isolate_id, drug, phenotype) -- R / S, with isolates missing
                     where untested
* lineages.csv     : (isolate_id, lineage)         -- metadata
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resistance catalogue: drug -> {mutation: penetrance}
# Penetrance ~ how strongly the mutation pushes toward a resistant phenotype.
# Values are illustrative but ordered to mirror real biology (e.g. rpoB S450L
# is a dominant rifampicin marker; pncA mutations are diffuse and weaker).
# ---------------------------------------------------------------------------
RESISTANCE_CATALOGUE: dict[str, dict[str, float]] = {
    # Rifampicin & isoniazid: dominated by single strong markers -> near-solved.
    "Rifampicin": {"rpoB_S450L": 6.5, "rpoB_H445Y": 5.0, "rpoB_D435V": 4.5, "rpoB_L452P": 3.0},
    "Isoniazid": {"katG_S315T": 6.2, "katG_S315N": 4.5, "inhA_c-15t": 3.2, "fabG1_g-17t": 2.6},
    # Ethambutol: weak individual effects but a strong epistatic pair (embB
    # M306V+G406A, often co-acquired) -> resistance is mostly an AND that a
    # LINEAR model structurally cannot represent but a tree can. This is the
    # case that justifies XGBoost over the logistic baseline.
    "Ethambutol": {"embB_M306V": 0.4, "embB_M306I": 1.6, "embB_G406A": 0.4, "embB_Q497R": 1.8},
    # Pyrazinamide: diffuse pncA mutations, no single dominant marker -> hardest.
    "Pyrazinamide": {"pncA_loss": 3.5, "pncA_H57D": 2.4, "pncA_T47A": 2.0, "pncA_V139A": 1.6},
    "Levofloxacin": {"gyrA_D94G": 6.0, "gyrA_D94N": 4.8, "gyrA_A90V": 3.6, "gyrB_E501D": 2.2},
    "Amikacin": {"rrs_a1401g": 6.5, "rrs_g1484t": 4.5, "eis_c-14t": 2.0},
    "Streptomycin": {"rpsL_K43R": 6.0, "rpsL_K88R": 4.2, "rrs_a514c": 3.0, "gid_loss": 2.0},
}

# Epistasis: pairs of mutations whose *joint* presence confers extra resistance
# beyond their additive effect. Linear models can't represent this; trees can.
INTERACTIONS: dict[str, list[tuple[tuple[str, str], float]]] = {
    "Ethambutol": [(("embB_M306V", "embB_G406A"), 7.0)],
}

# Base log-odds of resistance when NO causal mutation is present
# (captures unknown mechanisms; lower = rarer background resistance).
BASE_LOGIT = -3.2

# Background (lineage-defining + neutral) mutations -- pure noise features.
N_BACKGROUND_MUTATIONS = 90
LINEAGES = ["L1", "L2_Beijing", "L3", "L4", "L4.9"]

# How often each drug is actually tested (incomplete DST). First-line drugs are
# tested more often than second-line.
DST_COVERAGE = {
    "Rifampicin": 0.98, "Isoniazid": 0.98, "Ethambutol": 0.90,
    "Pyrazinamide": 0.80, "Levofloxacin": 0.65, "Amikacin": 0.55,
    "Streptomycin": 0.70,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_isolates: int = 4000,
    seed: int = 42,
    phenotype_noise: float = 0.03,   # prob. of a flipped DST label
    mdr_strength: float = 1.4,       # how strongly resistances co-occur
) -> dict[str, pd.DataFrame]:
    """Generate synthetic isolates and return {'variants','phenotypes','lineages'}."""
    rng = np.random.default_rng(seed)

    all_res_mutations = sorted({m for d in RESISTANCE_CATALOGUE.values() for m in d})
    background_mutations = [f"bg_snp_{i:03d}" for i in range(N_BACKGROUND_MUTATIONS)]

    # --- lineage assignment + lineage-specific background mutation profiles ----
    lineage = rng.choice(LINEAGES, size=n_isolates, p=[0.12, 0.30, 0.13, 0.30, 0.15])
    # Each lineage switches on a characteristic subset of background SNPs.
    lineage_profiles = {
        lin: rng.random(N_BACKGROUND_MUTATIONS) < rng.uniform(0.15, 0.35)
        for lin in LINEAGES
    }

    variant_rows: list[tuple[str, str]] = []
    pheno_rows: list[tuple[str, str, str]] = []
    lineage_rows: list[tuple[str, str]] = []

    # A per-isolate latent "resistance propensity" drives MDR clustering: an
    # isolate prone to one resistance is prone to others.
    propensity = rng.normal(0, 1, size=n_isolates)

    for i in range(n_isolates):
        iid = f"ISO_{i:05d}"
        lin = lineage[i]
        lineage_rows.append((iid, lin))

        present: set[str] = set()

        # Background mutations: lineage profile + a little individual noise.
        prof = lineage_profiles[lin]
        for j, bg in enumerate(background_mutations):
            if prof[j] or (rng.random() < 0.03):
                present.add(bg)

        # Resistance mutations: acquisition prob. rises with MDR propensity.
        acquire_logit = -2.4 + mdr_strength * propensity[i]
        p_acquire = _sigmoid(np.array([acquire_logit]))[0]
        for mut in all_res_mutations:
            # rarer mutations slightly less likely; keeps marginal freqs sane
            if rng.random() < p_acquire * rng.uniform(0.25, 0.75):
                present.add(mut)

        # Linked co-acquisition: embB_M306V and embB_G406A tend to occur together
        # (~half the time), which is what makes their epistatic AND learnable.
        if "embB_M306V" in present and rng.random() < 0.5:
            present.add("embB_G406A")

        for mut in sorted(present):
            variant_rows.append((iid, mut))

        # --- phenotypes per drug from a logistic model over present mutations --
        for drug, catalogue in RESISTANCE_CATALOGUE.items():
            logit = BASE_LOGIT
            for mut, penetrance in catalogue.items():
                if mut in present:
                    logit += penetrance
            # epistatic interactions (joint presence -> super-additive effect)
            for (mut_a, mut_b), bonus in INTERACTIONS.get(drug, []):
                if mut_a in present and mut_b in present:
                    logit += bonus
            p_res = _sigmoid(np.array([logit]))[0]
            resistant = rng.random() < p_res
            # phenotyping error
            if rng.random() < phenotype_noise:
                resistant = not resistant
            # incomplete testing
            if rng.random() < DST_COVERAGE[drug]:
                pheno_rows.append((iid, drug, "R" if resistant else "S"))

    variants = pd.DataFrame(variant_rows, columns=["isolate_id", "mutation"])
    phenotypes = pd.DataFrame(pheno_rows, columns=["isolate_id", "drug", "phenotype"])
    lineages = pd.DataFrame(lineage_rows, columns=["isolate_id", "lineage"])
    return {"variants": variants, "phenotypes": phenotypes, "lineages": lineages}


def write_sample(out_dir: str, **kwargs) -> None:
    """Generate and persist the sample CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    data = generate(**kwargs)
    for name, df in data.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  wrote {path}  ({len(df):,} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic TB-AMR data")
    parser.add_argument("--out", default="data/sample", help="output directory")
    parser.add_argument("--n", type=int, default=4000, help="number of isolates")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n:,} synthetic isolates -> {args.out}")
    write_sample(args.out, n_isolates=args.n, seed=args.seed)
