"""
TB-AMR decision-support demo (Streamlit).

Paste an isolate's variants or upload its VCF, and get a predicted resistance
profile across all 13 drugs with the per-call SHAP drivers behind each
prediction. Thin UI over src/serve/predict.py.

Run:  streamlit run app.py
"""
import os
import streamlit as st

from src.serve.predict import (
    load_models, load_thresholds, load_reliability,
    tokens_from_text, tokens_from_vcf_bytes, predict_isolate,
)

st.set_page_config(page_title="TB-AMR Predictor", page_icon="🧬", layout="wide")

MODELS_DIR = os.environ.get("TBAMR_MODELS", "models")
RELIABLE_AUC = 0.85          # below this, a drug's model is flagged low-confidence

EXAMPLES = {
    "— choose —": "",
    "Susceptible (no resistance markers)": "g3844992_T>A",
    "MDR — rifampicin + isoniazid": "rpoB@761155_C>T\nkatG@2155168_C>G",
    "Pre-XDR — add a fluoroquinolone (gyrA D94)":
        "rpoB@761155_C>T\nkatG@2155168_C>G\ngyrA@7581_GACAG>GGCAC",
}


@st.cache_resource
def _load():
    models, feats = load_models(MODELS_DIR)
    return models, feats, load_thresholds("reports"), load_reliability("reports")


st.title("🧬 TB drug-resistance predictor")
st.caption("Genome-based resistance prediction for *M. tuberculosis*, trained on "
           "12,288 CRyPTIC isolates. Research demo — not a clinical device.")

try:
    models, feats, thr90, reliability = _load()
except FileNotFoundError:
    st.error(f"No models found in `{MODELS_DIR}/`. Run `python run_pipeline.py "
             "--data data/vcf_indel` first, or set TBAMR_MODELS.")
    st.stop()

has_burden = any(f.startswith("burden::") for f in feats)

# ---- sidebar: decision-threshold policy -------------------------------------
mode = st.sidebar.radio(
    "Decision threshold",
    ["Balanced (0.5)", "High-sensitivity (catch ~90% of resistance)"],
    index=0,
)
st.sidebar.caption(
    "**Balanced** favours specificity — fewer false alarms.\n\n"
    "**High-sensitivity** uses per-drug thresholds tuned to catch ~90% of "
    "resistant isolates, at the cost of many more false positives — especially "
    "for the rare last-line drugs (their thresholds are near zero)."
)
use_thr90 = mode.startswith("High")
thresholds = thr90 if (use_thr90 and thr90) else {}

st.success(f"Loaded {len(models)} per-drug models · {len(feats):,} features "
           f"({'SNP + gene-burden' if has_burden else 'SNP'}) · "
           f"threshold: {'90% sensitivity' if thresholds else 'balanced (0.5)'}.")

if "variant_text" not in st.session_state:
    st.session_state.variant_text = ""

left, right = st.columns([1, 1])
with left:
    st.subheader("Isolate variants")
    ex = st.selectbox("Load an example", list(EXAMPLES))
    if ex != "— choose —" and st.button("Use this example"):
        st.session_state.variant_text = EXAMPLES[ex]
    st.session_state.variant_text = st.text_area(
        "Paste variant tokens (one per line)",
        value=st.session_state.variant_text, height=180,
        placeholder="rpoB@761155_C>T\nkatG@2155168_C>G\nor raw  NC_000962.3_761155_C_T",
    )
    up = st.file_uploader("…or upload a VCF (.vcf / .vcf.gz)", type=["vcf", "gz"])
    go = st.button("Predict resistance profile", type="primary")

with right:
    st.subheader("How it works")
    st.markdown(
        "- Variants become a binary feature vector"
        f"{' plus per-gene burden counts' if has_burden else ''}, exactly as in "
        "training.\n"
        "- Each drug's XGBoost model outputs a resistance probability; the call "
        "uses the threshold selected in the sidebar.\n"
        "- SHAP shows which variants drove each prediction (🔺 toward resistant, "
        "🔻 toward susceptible).\n"
        f"- Drugs with cross-validated AUC below {RELIABLE_AUC:.2f} are flagged "
        "**low-confidence** — the model is too weak there to act on."
    )

if go:
    tokens = set()
    if up is not None:
        tokens |= tokens_from_vcf_bytes(up.getvalue())
    tokens |= tokens_from_text(st.session_state.variant_text)
    if not tokens:
        st.warning("No variants provided.")
        st.stop()

    results = predict_isolate(models, feats, tokens, thresholds)

    def _reliable(drug):
        auc = reliability.get(drug)
        return (auc is None) or (auc >= RELIABLE_AUC)

    n_r = sum(r["call"] == "R" and _reliable(r["drug"]) for r in results)

    st.divider()
    st.markdown(f"### Profile — predicted resistant to **{n_r} of {len(results)}** "
                "drugs *(confident calls)*")
    st.caption(f"{len(tokens)} variant(s) recognised.")

    for r in results:
        auc = reliability.get(r["drug"])
        low = auc is not None and auc < RELIABLE_AUC
        c1, c2, c3 = st.columns([2, 1, 3])
        c1.markdown(f"**{r['drug'].title()}**")
        if low:
            c1.caption(f"⚠ low-confidence model (AUC {auc:.2f})")
        c2.markdown(":red[**RESISTANT**]" if r["call"] == "R"
                    else ":green[Susceptible]")
        c3.progress(min(max(r["prob"], 0.0), 1.0),
                    text=f"p = {r['prob']:.2f}  (threshold {r['threshold']:.2f})")
        if r["drivers"]:
            with c3.expander("why this call"):
                for d in r["drivers"]:
                    arrow = "🔺" if d["contrib"] > 0 else "🔻"
                    state = "present" if d["present"] else "absent"
                    st.write(f"{arrow} `{d['feature']}` — {state}, "
                             f"SHAP {d['contrib']:+.2f}")

    st.divider()
    st.caption("Predictions are model estimates from genotype alone and do not "
               "replace phenotypic drug-susceptibility testing. Last-line drugs "
               "(bedaquiline, linezolid, clofazimine, delamanid) are low-confidence "
               "by nature — see the project results.")
