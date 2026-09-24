"""RiskLoop Portfolio Streamlit Demo Application.

SRD Traceability: Portfolio Interface, FR-14..FR-16, FR-21, FR-22, NFR-06..NFR-08
"""

import sys
import html
from pathlib import Path
import streamlit as st

# Add src/ to sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from riskloop.inference.predictor import RiskLoopPredictor
from riskloop.inference.loader import REPRESENTATIVE_RUN_MAPPING

# Page Configuration
st.set_page_config(
    page_title="RiskLoop - AI Contract Risk Extraction",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-detected {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .card-not-detected {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .badge-detected {
        background-color: #166534;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .badge-not-detected {
        background-color: #64748B;
        color: #FFFFFF;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .score-label {
        font-size: 0.9rem;
        color: #475569;
        margin-top: 0.5rem;
    }
    .score-val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
    }
    .context-box {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 0.8rem 1rem;
        font-family: monospace;
        font-size: 0.9rem;
        border-radius: 4px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .highlight-span {
        background-color: #FEF08A;
        color: #854D0E;
        font-weight: 600;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .disclaimer-footer {
        font-size: 0.8rem;
        color: #94A3B8;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading RiskLoop models...")
def get_predictor() -> RiskLoopPredictor:
    """Instantiate and cache RiskLoopPredictor resource."""
    return RiskLoopPredictor(pretrained=True)


def load_sample_contract(filename: str) -> str:
    """Load text content of a sample contract file."""
    sample_path = Path(__file__).parent / "examples" / filename
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def main():
    # Header Section
    st.markdown('<div class="main-header">⚖️ RiskLoop: Contract Risk Extraction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Analyze legal contracts for target risk clauses using frozen, representative Legal-BERT models.</div>',
        unsafe_allow_html=True
    )

    # Sidebar Options
    st.sidebar.header("Input Contract Options")
    input_method = st.sidebar.radio(
        "Choose Input Method",
        ["Sample Contract", "Upload .txt File", "Paste Text"],
        index=0
    )

    contract_text = ""
    source_name = "Custom Input"

    if input_method == "Sample Contract":
        sample_choice = st.sidebar.selectbox(
            "Select Sample Contract",
            ["NDA Demo (Non-Disclosure Agreement)", "Services Agreement Demo (Master Services)"]
        )
        if "NDA Demo" in sample_choice:
            contract_text = load_sample_contract("sample_contract_nda.txt")
            source_name = "Sample NDA Demo"
        else:
            contract_text = load_sample_contract("sample_contract_services.txt")
            source_name = "Sample Services Agreement Demo"

    elif input_method == "Upload .txt File":
        uploaded_file = st.sidebar.file_uploader("Upload Contract Text File", type=["txt"])
        if uploaded_file is not None:
            contract_text = uploaded_file.read().decode("utf-8")
            source_name = uploaded_file.name

    elif input_method == "Paste Text":
        contract_text = st.sidebar.text_area("Paste Contract Text Here", height=250)
        source_name = "Pasted Contract Text"

    # Contract Text Preview Section
    with st.expander("📄 View Input Contract Text", expanded=(input_method != "Sample Contract")):
        if contract_text.strip():
            st.text_area("Contract Content", contract_text, height=200, disabled=True)
            st.caption(f"Length: {len(contract_text):,} characters")
        else:
            st.info("No contract text loaded yet. Please select a sample contract, upload a .txt file, or paste contract text.")

    # Action Button
    analyze_clicked = st.button("🔍 Analyze Contract", type="primary", use_container_width=True)

    # Maintain prediction in session state
    if analyze_clicked:
        if not contract_text or not contract_text.strip():
            st.error("Please select or enter contract text before running analysis.")
            return

        try:
            with st.spinner("Executing 512-token sliding window inference..."):
                predictor = get_predictor()
                st.session_state["prediction"] = predictor.predict(contract_text)
                st.session_state["analyzed_source"] = source_name
                st.session_state["analyzed_text"] = contract_text
        except Exception as e:
            st.error(f"Inference Exception: {str(e)}")
            st.exception(e)
            return

    # Render Prediction Results
    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        orig_text = st.session_state["analyzed_text"]
        analyzed_name = st.session_state["analyzed_source"]

        st.markdown("---")
        st.subheader("📊 Analysis Summary")

        tasks = pred["tasks"]
        detected_count = sum(1 for t in tasks.values() if t["detected"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Contract Analyzed", analyzed_name)
        col2.metric("Length", f"{pred['contract_length_chars']:,} chars")
        col3.metric("Chunks Processed", pred["num_chunks"])
        col4.metric("Target Clauses Detected", f"{detected_count} / {len(tasks)}")

        st.caption("ℹ️ *Model score is an uncalibrated logit-based detection score; it is not a probability.*")

        st.markdown("### 🎯 Risk Clause Detection Results")

        cards_cols = st.columns(3)
        for idx, (task_name, task_data) in enumerate(tasks.items()):
            with cards_cols[idx]:
                st.markdown(f"#### {task_name}")

                if task_data["detected"]:
                    st.markdown('<span class="badge-detected">DETECTED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-not-detected">NOT DETECTED</span>', unsafe_allow_html=True)

                st.markdown(f'<div class="score-label">Model Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score-val">{task_data["model_score"]:+.6f}</div>', unsafe_allow_html=True)

                st.caption(f"Representative: **{task_data['model']}** (Seed {task_data['seed']})")

                if task_data["detected"] and task_data["character_start"] is not None:
                    c_s = task_data["character_start"]
                    c_e = task_data["character_end"]
                    st.caption(f"Character Offsets: `[{c_s} .. {c_e}]`")
                    st.markdown("**Evidence Snippet:**")
                    st.info(f'"{task_data["predicted_text"]}"')
                else:
                    st.caption("Evidence: *Unavailable / None*")

        # Evidence & Document Context Viewer
        st.markdown("---")
        st.markdown("### 🔍 Detected Clause Evidence & Document Context")

        has_any_detected = any(t["detected"] for t in tasks.values())
        if not has_any_detected:
            st.info("No target risk clauses were detected in this document.")
        else:
            for task_name, task_data in tasks.items():
                if task_data["detected"]:
                    with st.expander(f"📍 Clause: {task_name}", expanded=True):
                        c_s = task_data["character_start"]
                        c_e = task_data["character_end"]
                        evidence = task_data["predicted_text"]

                        if c_s is not None and c_e is not None and evidence:
                            # Extract surrounding context (up to 200 chars before and after)
                            ctx_start = max(0, c_s - 200)
                            ctx_end = min(len(orig_text), c_e + 200)

                            before_text = html.escape(orig_text[ctx_start:c_s])
                            evidence_text = html.escape(evidence)
                            after_text = html.escape(orig_text[c_e:ctx_end])

                            formatted_html = (
                                f'<div class="context-box">'
                                f'{before_text}'
                                f'<span class="highlight-span">{evidence_text}</span>'
                                f'{after_text}'
                                f'</div>'
                            )

                            st.markdown("**Extracted Evidence Text:**")
                            st.write(f"> *\"{evidence}\"*")
                            st.caption(f"Character offsets: `[{c_s} .. {c_e}]` | Length: {c_e - c_s} chars")

                            st.markdown("**Surrounding Document Context:**")
                            st.markdown(formatted_html, unsafe_allow_html=True)
                        else:
                            st.warning("Evidence text marked detected, but offset boundaries were unavailable.")

    # Technical Details Section
    with st.expander("🛠️ Technical Architecture & Pipeline Details", expanded=False):
        st.markdown("""
        **System Architecture:**
        - **Model Backbone:** `nlpaueb/legal-bert-base-uncased` (110M parameters)
        - **Inference Pipeline:** 512-token sliding window chunking with 256-token stride
        - **Offset Mapping:** Character-level span resolution via Hugging Face `return_offsets_mapping=True`
        - **Target Tasks & Representative Models:**
          - **Cap On Liability:** `run_03` (Seed 44)
          - **Anti-Assignment:** `run_04` (Seed 42)
          - **Termination For Convenience:** `run_07` (Seed 42)
        - **Logit Decoding:** Standardized $S_{\\text{span}} - S_{\\text{no\_answer}}$ logit difference over valid document text tokens.
        """)

    # Frozen Phase 6 Test Performance Section
    with st.expander("📊 Frozen Phase 6 Test Set Performance", expanded=False):
        st.markdown("Official Phase 6 single-pass test evaluation results across 150 held-out test set contracts:")

        metrics_data = [
            {
                "Task": "Anti-Assignment",
                "Test F1": "0.808362",
                "ROC-AUC": "0.994447",
                "PR-AUC": "0.914671",
                "Representative Model": "run_04 (Seed 42)"
            },
            {
                "Task": "Cap On Liability",
                "Test F1": "0.772727",
                "ROC-AUC": "0.996049",
                "PR-AUC": "0.914772",
                "Representative Model": "run_03 (Seed 44)"
            },
            {
                "Task": "Termination For Convenience",
                "Test F1": "0.748387",
                "ROC-AUC": "0.993907",
                "PR-AUC": "0.792695",
                "Representative Model": "run_07 (Seed 42)"
            }
        ]
        st.table(metrics_data)
        st.caption("🔒 *These metrics were produced during locked Phase 6 evaluation on frozen test contracts.*")

    # Disclaimer Footer
    st.markdown(
        '<div class="disclaimer-footer">'
        'RiskLoop is a portfolio ML project demonstrating contract clause detection using Legal-BERT. '
        'Results are model predictions and are not legal advice.'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
