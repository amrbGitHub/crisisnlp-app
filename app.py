"""
CrisisNLP - risk-band screening prototype
PROG74040 Group 6, Phase 2 deployment.

Loads the fine-tuned mental-roberta-base checkpoint from the Hugging Face Hub,
applies INT8 dynamic quantization for CPU inference, and returns a 4-class risk
band with token-level attribution.

This is a research prototype, not a diagnostic instrument. See the standing
notice in the sidebar.
"""

import gc
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_REPO = os.environ.get("MODEL_REPO", "incodev/crisisnlp-risk-roberta")
LABELS = ["No Risk", "Low Risk", "Moderate Risk", "High Risk"]
HIGH_RISK = 3
MAX_LEN = 256

# Muted, monotone-to-warm scale. Deliberately not a red/green traffic light:
# this output is a prompt for a human to look, not a verdict.
BAND_COLOR = {0: "#5A7D9A", 1: "#6E8B7B", 2: "#C08A4B", 3: "#A8504F"}

st.set_page_config(page_title="CrisisNLP - risk screening", page_icon="◐", layout="wide")

st.markdown(
    """
    <style>
      .band-card {padding: 1.1rem 1.3rem; border-radius: 10px; color: #fff; margin-bottom: 0.6rem;}
      .band-card h2 {margin: 0; font-size: 1.55rem; font-weight: 600; letter-spacing: -0.01em;}
      .band-card p {margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.9rem;}
      .tok {padding: 1px 3px; border-radius: 3px;}
      .muted {color: #6b7280; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model (first run downloads ~500 MB)...")
def load_model(repo_id: str, quantize: bool = True):
    token = st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=token)
    model.eval()

    cfg = {}
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(repo_id, "model_config.json", token=token)) as f:
            cfg = json.load(f)
    except Exception:
        pass

    if quantize:
        # The quantization entry point moved between torch versions; try both.
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:
            from torch.quantization import quantize_dynamic
        try:
            import gc
            quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            del model
            gc.collect()
            model = quantized
            cfg["_quantized"] = True
        except Exception as exc:
            cfg["_quantized"] = False
            cfg["_quant_error"] = f"{type(exc).__name__}: {exc}"
    return tokenizer, model, cfg


@torch.no_grad()
def predict_proba(texts, tokenizer, model, batch_size: int = 4) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            list(texts[i : i + batch_size]),
            truncation=True,
            padding="longest",     # pad to the longest post in the batch, not to MAX_LEN
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        out.append(torch.softmax(model(**batch).logits, dim=-1).numpy())
        del batch
    return np.vstack(out)


def apply_bias(probs: np.ndarray, bias: float) -> np.ndarray:
    """Reproduce the notebook's calibration in probability space."""
    logits = np.log(np.clip(probs, 1e-9, None))
    logits[:, HIGH_RISK] += bias
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Sidebar: standing safety notice + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Before you use this")
    st.warning(
        "This tool estimates a risk band from writing style and word choice. "
        "It was trained on public Reddit posts with labels the project team "
        "constructed, not labels assigned by clinicians. It cannot diagnose "
        "anyone and it should never be the only basis for a decision about a "
        "person."
    )
    st.markdown(
        "**If you or someone you know needs help now**\n\n"
        "- Canada and US: call or text **988**\n"
        "- UK and Ireland: call **116 123** (Samaritans)\n"
        "- Anywhere else: [findahelpline.com](https://findahelpline.com)\n\n"
        "If someone is in immediate danger, contact emergency services."
    )
    st.divider()

    st.subheader("Settings")
    repo_id = st.text_input("Model repository", DEFAULT_REPO)
    quantize = st.checkbox("INT8 quantization", value=True, help="Faster on CPU, ~4x smaller in memory.")
    st.caption("Sensitivity shifts the High-Risk decision boundary. Higher catches more "
               "High-Risk posts and raises the false-alarm rate.")
    bias = st.slider("High-Risk sensitivity", 0.0, 4.0, 1.15, 0.05)
    show_lime = st.checkbox("Explain the prediction", value=True,
                            help="Adds roughly 20-40 seconds per post on CPU.")

    st.divider()
    st.subheader("Latency")
    lat = st.session_state.get("latencies", [])
    if lat:
        warm = lat[1:] if len(lat) > 1 else lat   # discard the cold first call
        med = float(np.median(warm))
        st.metric("Median inference", f"{med:.0f} ms",
                  help="Model forward pass only, excluding page rendering and LIME.")
        st.caption(f"{len(warm)} warm run(s). Phase 1 target: under 300 ms.")
        if st.button("Reset timings"):
            st.session_state["latencies"] = []
    else:
        st.caption("Screen a post to start measuring.")

    with st.expander("Run a benchmark"):
        st.caption("Scores the same post repeatedly and reports the median, "
                   "which is more stable than a single measurement on shared CPU.")
        n_runs = st.number_input("Runs", 3, 15, 7, 1)
        if st.button("Run benchmark"):
            sample = ("i have been feeling completely overwhelmed lately and i do not know "
                      "who to talk to about any of it anymore")
            times = []
            bar = st.progress(0.0)
            for i in range(int(n_runs)):
                t = time.perf_counter()
                predict_proba([sample], tokenizer, model)
                times.append((time.perf_counter() - t) * 1000.0)
                gc.collect()
                bar.progress((i + 1) / float(n_runs))
            warm = times[1:] if len(times) > 1 else times
            st.session_state["bench"] = {
                "n": len(warm), "median": float(np.median(warm)),
                "min": float(np.min(warm)), "max": float(np.max(warm)),
            }
        b = st.session_state.get("bench")
        if b:
            st.write(
                f"**Median {b['median']:.0f} ms** across {b['n']} warm runs "
                f"(range {b['min']:.0f} to {b['max']:.0f} ms)."
            )
            st.caption("Free-tier CPU allocation varies, so report the median, not a single run.")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("CrisisNLP")
st.caption("Risk-band screening for social media text - PROG74040 Group 6, Phase 2 prototype")

try:
    tokenizer, model, cfg = load_model(repo_id, quantize)
except Exception as exc:
    st.error(
        f"Could not load `{repo_id}`. Check the repository name in the sidebar, and if the "
        f"repo is private, add an `HF_TOKEN` under Streamlit secrets.\n\n```\n{exc}\n```"
    )
    st.stop()

if quantize and cfg.get("_quantized") is False:
    st.warning(
        "INT8 quantization did not apply on this runtime, so the model is running in FP32 "
        f"and using roughly twice the memory. Cause: {cfg.get('_quant_error', 'unknown')}"
    )

tuned_bias = cfg.get("high_risk_logit_bias")
if tuned_bias is not None and abs(bias - tuned_bias) > 0.01:
    st.info(
        f"Sensitivity is at {bias:.2f}. The calibrated deployment setting reported in the "
        f"write-up is **{tuned_bias:.2f}** - set the sidebar slider there to reproduce the "
        "documented results."
    )

tab_single, tab_batch, tab_model = st.tabs(["Single post", "Batch file", "Model card"])

# ---------------------------- Single post ---------------------------------
with tab_single:
    text = st.text_area(
        "Post text",
        height=180,
        placeholder="Paste a social media post to screen.",
    )
    go = st.button("Screen this post", type="primary")

    if go and text.strip():
        t0 = time.perf_counter()
        probs = predict_proba([text], tokenizer, model)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if bias:
            probs = apply_bias(probs, bias)
        p = probs[0]
        band = int(p.argmax())
        st.session_state.setdefault("latencies", []).append(infer_ms)

        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                f"<div class='band-card' style='background:{BAND_COLOR[band]}'>"
                f"<h2>{LABELS[band]}</h2>"
                f"<p>Confidence {p[band]:.0%} &nbsp;·&nbsp; inference {infer_ms:.0f} ms</p></div>",
                unsafe_allow_html=True,
            )
            if band == HIGH_RISK:
                st.error(
                    "Flagged for human review. Route this to a trained responder rather than "
                    "acting on the label directly."
                )
            st.dataframe(
                pd.DataFrame({"Risk band": LABELS, "Probability": p})
                .style.format({"Probability": "{:.1%}"})
                .bar(subset=["Probability"], color="#d5dce4"),
                hide_index=True,
                use_container_width=True,
            )

        with right:
            wc = len(text.split())
            st.metric("Words", wc)
            if wc < 15:
                st.caption(
                    "Short posts carry little signal and the model is least reliable on them. "
                    "Treat this band as weak evidence."
                )
            if wc > 200:
                st.caption(f"Only the first {MAX_LEN} tokens are read; the rest is truncated.")

        if show_lime:
            with st.spinner("Working out which words drove this..."):
                try:
                    from lime.lime_text import LimeTextExplainer

                    explainer = LimeTextExplainer(class_names=LABELS)
                    exp = explainer.explain_instance(
                        text,
                        lambda xs: predict_proba(xs, tokenizer, model),
                        num_features=10,
                        labels=[band],
                        num_samples=200,
                    )
                    st.subheader("What drove this prediction")
                    st.caption(
                        "LIME re-scores the post with words removed to see which ones move the "
                        "prediction. It explains this one post, not the model as a whole."
                    )
                    pairs = exp.as_list(label=band)
                    ex_df = pd.DataFrame(pairs, columns=["Token", "Contribution"])
                    ex_df["Direction"] = np.where(
                        ex_df.Contribution > 0, f"toward {LABELS[band]}", f"away from {LABELS[band]}"
                    )
                    st.dataframe(
                        ex_df.style.format({"Contribution": "{:+.3f}"}).bar(
                            subset=["Contribution"], align="zero", color=["#8fb0c9", "#c9998f"]
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                except ImportError:
                    st.info("Install `lime` to see token attribution.")
    elif go:
        st.warning("Enter some text first.")

# ---------------------------- Batch file ----------------------------------
with tab_batch:
    st.write("Upload a CSV with a `text` column to screen several posts at once.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        col = "text" if "text" in df.columns else ("text_clean" if "text_clean" in df.columns else None)
        if col is None:
            st.error("No `text` or `text_clean` column found in that file.")
        else:
            n = min(len(df), 50)
            if len(df) > n:
                st.caption(f"Screening the first {n} of {len(df)} rows to keep this responsive.")
            df = df.head(n).copy()
            with st.spinner(f"Screening {n} posts..."):
                probs = predict_proba(df[col].astype(str).tolist(), tokenizer, model)
                if bias:
                    probs = apply_bias(probs, bias)
            df["predicted_band"] = [LABELS[i] for i in probs.argmax(1)]
            df["confidence"] = probs.max(1).round(3)
            df["p_high_risk"] = probs[:, HIGH_RISK].round(3)

            counts = df.predicted_band.value_counts().reindex(LABELS).fillna(0).astype(int)
            st.bar_chart(counts)
            st.dataframe(df.sort_values("p_high_risk", ascending=False), use_container_width=True)
            st.download_button(
                "Download predictions",
                df.to_csv(index=False).encode(),
                "crisisnlp_predictions.csv",
                "text/csv",
            )

# ---------------------------- Model card ----------------------------------
with tab_model:
    st.subheader("What this model is")
    st.markdown(
        "`mental/mental-roberta-base` fine-tuned on a 7,122-post unified dataset assembled from "
        "three public corpora: **dreaddit**, **combined-set**, and **Suicide_Detection**."
    )
    if cfg:
        c1, c2 = st.columns(2)
        cal = cfg.get("test_calibrated", {})
        arg = cfg.get("test_argmax", {})
        with c1:
            st.metric("Test weighted F1", f"{cal.get('weighted_f1', arg.get('weighted_f1', 0)):.3f}",
                      help="Proposal target: 0.82")
        with c2:
            st.metric("High-Risk recall", f"{cal.get('high_risk_recall', arg.get('high_risk_recall', 0)):.3f}",
                      help="Proposal target: 0.90")

    st.subheader("Known limitations")
    st.markdown(
        "- The four risk bands are a **severity ladder the team constructed**, not clinician-graded "
        "labels. No source corpus carries a native 4-class label.\n"
        "- No Risk and Low Risk come only from dreaddit. Moderate Risk and High Risk come only from "
        "combined-set and Suicide_Detection. The model therefore learns some amount of "
        "*which corpus a post came from* alongside genuine risk signal.\n"
        "- Training data is public English-language Reddit text. Performance on other platforms, "
        "other registers, or other languages is unmeasured.\n"
        "- Nothing here was validated against clinical outcomes."
    )
    st.subheader("Intended use")
    st.markdown(
        "Surfacing posts for a human to read, in a research setting. Not for triage decisions, "
        "not for anything involving an identifiable person without their consent, and not as "
        "evidence about anyone's mental state."
    )
    if cfg.get("limitations"):
        with st.expander("Limitations recorded at training time"):
            for line in cfg["limitations"]:
                st.write("-", line)
