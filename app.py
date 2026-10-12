"""Medical Vision AI: Production-Structured Computer Vision POC.

Streamlit web application demonstrating real deep learning inference on
chest X-ray images using a medically pretrained TorchXRayVision DenseNet-121.
"""

import io
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.config import AppConfig, load_config
from src.history_repository import InMemoryHistoryRepository
from src.inference import run_inference, InferenceExecutionError
from src.model_loader import load_chest_xray_model, ModelLoadError
from src.preprocessing import preprocess_chest_xray
from src.schemas import ImageMetadata, InferenceResult
from src.utils import format_duration_ms, format_score_percentage
from src.validation import validate_image_bytes

# Page configuration
st.set_page_config(
    page_title="Medical Vision AI — Chest X-Ray Classifier",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom clinical UI styles
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    
    /* Disclaimer banner */
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #f59e0b;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Metric pill */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    /* Pathology pill */
    .finding-detected {
        background: rgba(239, 68, 68, 0.15);
        border-left: 4px solid #ef4444;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .finding-normal {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    
    /* Custom table header styling */
    .stTable th {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_config() -> AppConfig:
    """Load and cache application configuration."""
    return load_config()


@st.cache_resource(show_spinner="Loading pretrained medical computer vision model weights...")
def get_cached_model():
    """Load and cache the TorchXRayVision DenseNet-121 model."""
    config = get_cached_config()
    return load_chest_xray_model(config.model)


# Initialize in-memory session history if not already initialized
if "session_repo" not in st.session_state:
    st.session_state["session_repo"] = InMemoryHistoryRepository()

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


def render_sidebar(config: AppConfig):
    """Render technical details, sample images, and session control in the sidebar."""
    with st.sidebar:
        st.markdown("### 🩻 Medical Vision AI")
        st.markdown(
            "Production-structured computer vision proof-of-concept demonstrating "
            "actual deep learning inference on chest radiographs."
        )

        st.markdown("---")
        st.markdown("#### 🔬 Verified Model Card")
        st.markdown(f"**Architecture:** `{config.model.architecture}` (DenseNet-121)")
        st.markdown(f"**Weights:** `{config.model.weights}`")
        st.markdown(f"**Provider:** `{config.model.provider}` (v{config.model.library_version})")
        st.markdown(f"**License:** `{config.model.license}`")
        st.markdown(f"**Pretraining Datasets:** NIH ChestX-ray14, CheXpert, PadChest, MIMIC-CXR, OpenI")
        st.markdown(f"**Input Norm:** `[-1024, 1024]` @ `224×224`")
        st.markdown(f"**Execution Device:** `{config.model.device.upper()}`")

        with st.expander("📋 Supported Pathologies (18 Classes)"):
            st.markdown(
                "\n".join([f"- **{p}**" for p in config.pathologies])
            )

        st.markdown("---")
        st.markdown("#### 📂 Quick Sample Images")
        sample_dir = Path("data/sample_images")
        sample_options = {"Upload your own image": None}
        if sample_dir.exists():
            for f in sorted(sample_dir.glob("*.*")):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    clean_name = f.stem.replace("_", " ").title()
                    sample_options[f"Sample: {clean_name}"] = f

        chosen_sample = st.selectbox(
            "Select verification sample:",
            options=list(sample_options.keys()),
            index=0,
            help="Select a bundled sample image from the official NIH dataset to test inference instantly.",
        )

        st.markdown("---")
        st.markdown("#### ⏱️ Session Demo History")
        repo: InMemoryHistoryRepository = st.session_state["session_repo"]
        history_count = repo.count()
        st.markdown(f"Total analyses this session: **{history_count}**")

        if history_count > 0:
            if st.button("🗑️ Clear Session History", use_container_width=True):
                repo.clear()
                st.session_state["last_result"] = None
                st.rerun()

        st.markdown("---")
        st.caption(
            "Portfolio POC • Built with PyTorch, TorchXRayVision, and Streamlit. "
            "Strictly stateless: images are not saved to disk."
        )

        return sample_options[chosen_sample]


def main():
    config = get_cached_config()
    selected_sample_file = render_sidebar(config)

    # Header section
    st.markdown("## 🩻 Medical Vision AI — Chest Radiography Classifier")
    st.markdown(
        "Real-time pathology screening on chest X-rays using a medically pretrained **DenseNet-121** model. "
        "Every score is derived from actual forward-pass tensor computation."
    )

    # Prominent medical research disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <span>⚠️</span>
            <div>
                <strong>PORTFOLIO RESEARCH DEMO — NOT FOR CLINICAL USE:</strong>
                This application is an AI developer portfolio demonstration. It does not provide medical advice,
                has not undergone clinical validation, and must never be used for medical diagnostic or treatment decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Layout: Upload & preview columns
    col_input, col_preview = st.columns([1, 1], gap="medium")

    uploaded_bytes = None
    uploaded_name = None

    with col_input:
        st.markdown("### 1. Image Upload")
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray (PNG or JPEG, max 10MB)",
            type=["png", "jpg", "jpeg"],
            help="Upload a standard frontal chest X-ray image (PA or AP view).",
        )

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_name = uploaded_file.name
        elif selected_sample_file is not None:
            st.info(f"Using bundled test image: **{selected_sample_file.name}**")
            with open(selected_sample_file, "rb") as f:
                uploaded_bytes = f.read()
            uploaded_name = selected_sample_file.name

        # Validate image if present
        val_result = None
        if uploaded_bytes is not None:
            val_result = validate_image_bytes(uploaded_bytes, uploaded_name, config.validation)

            if not val_result.is_valid:
                st.error(f"❌ Upload Validation Failed: {val_result.error_message}")
                return

            meta = val_result.image_metadata
            st.success(
                f"✅ Image validated successfully: **{meta.width}×{meta.height}** ({meta.format}, "
                f"{meta.size_bytes / 1024:.1f} KB, {meta.channels} channel(s))"
            )

            # Analyze action buttons
            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                run_btn = st.button("🚀 Analyze Chest X-ray", type="primary", use_container_width=True)
            with btn_col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state["last_result"] = None
                    st.rerun()

    with col_preview:
        st.markdown("### 2. Radiograph Preview")
        if uploaded_bytes is not None and val_result is not None and val_result.is_valid:
            pil_image = Image.open(io.BytesIO(uploaded_bytes))
            st.image(pil_image, caption=f"Uploaded Scan: {uploaded_name}", use_container_width=True)
        else:
            st.info("Upload an image or select a sample from the sidebar to preview.")

    # Execution logic
    if uploaded_bytes is not None and val_result is not None and val_result.is_valid:
        if run_btn:
            try:
                # 1. Load model
                model = get_cached_model()

                # 2. Preprocess
                with st.spinner("Preprocessing image to [-1024, 1024] TorchXRayVision tensor..."):
                    tensor = preprocess_chest_xray(uploaded_bytes, config.model)

                # 3. Inference
                with st.spinner("Executing model forward pass (evaluation mode)..."):
                    result = run_inference(tensor, model, val_result.image_metadata, config)

                # Store in session state and in-memory history repository
                st.session_state["last_result"] = result
                st.session_state["session_repo"].add(result)

            except ModelLoadError as e:
                st.error(f"❌ Model Loading Blocker: {str(e)}")
                return
            except InferenceExecutionError as e:
                st.error(f"❌ Model Inference Failure: {str(e)}")
                return
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                return

    # Render results if available
    current_result: Optional[InferenceResult] = st.session_state.get("last_result")
    if current_result is not None:
        st.markdown("---")
        st.markdown("### 3. Model Inference Results")

        # Top summary metric row
        m1, m2, m3, m4 = st.columns(4)
        top_finding = current_result.top_findings[0] if current_result.top_findings else None

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{format_duration_ms(current_result.inference_time_ms)}</div>
                    <div class="metric-label">Execution Latency</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {'#ef4444' if top_finding and top_finding.is_detected else '#10b981'};">
                        {top_finding.pathology if top_finding else 'None'}
                    </div>
                    <div class="metric-label">Primary Finding</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{format_score_percentage(top_finding.calibrated_score) if top_finding else '0%'}</div>
                    <div class="metric-label">Calibrated Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{len(current_result.predictions)}</div>
                    <div class="metric-label">Classes Evaluated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Findings columns: Left = Top Detected Findings Cards, Right = Calibrated Score Chart
        col_findings, col_chart = st.columns([1, 1], gap="medium")

        with col_findings:
            st.markdown("#### 🎯 Top Ranked Findings")
            st.caption(
                "Calibrated scores are normalized by pathology operating thresholds. "
                "A calibrated score ≥ 50.0% indicates a finding detected above the operating point."
            )

            for p in current_result.top_findings:
                status_class = "finding-detected" if p.is_detected else "finding-normal"
                status_badge = (
                    "🔴 **DETECTED (Above Threshold)**"
                    if p.is_detected
                    else "🟢 **Below Decision Threshold**"
                )

                st.markdown(
                    f"""
                    <div class="{status_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.05rem; font-weight:600;">{p.pathology}</span>
                            <span style="font-size:1.1rem; font-weight:700;">{format_score_percentage(p.calibrated_score)}</span>
                        </div>
                        <div style="font-size:0.82rem; color:#94a3b8; margin-top:4px;">
                            {status_badge} • Prob: <code>{p.probability:0.4f}</code> • Op Thresh: <code>{p.operating_threshold:0.4f}</code>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_chart:
            st.markdown("#### 📊 Multi-Pathology Distribution")
            st.caption("Distribution of calibrated operating-point scores across top evaluated pathologies:")

            chart_df = pd.DataFrame(
                [
                    {
                        "Pathology": p.pathology,
                        "Calibrated Score (%)": round(p.calibrated_score * 100, 1),
                    }
                    for p in current_result.top_findings
                ]
            ).set_index("Pathology")

            st.bar_chart(chart_df, color="#38bdf8", use_container_width=True)

        # Full 18-Pathology Diagnostic Table (Expandable)
        with st.expander("📑 Complete 18-Pathology Diagnostic Matrix", expanded=False):
            st.caption(
                "Complete breakdown of raw classifier outputs, uncalibrated sigmoid probabilities, "
                "and operating-point calibrated scores for all 18 supported chest pathologies."
            )

            table_rows = []
            for p in current_result.predictions:
                table_rows.append(
                    {
                        "Pathology": p.pathology,
                        "Status": "Detected (>Threshold)" if p.is_detected else "Below Threshold",
                        "Calibrated Score": format_score_percentage(p.calibrated_score),
                        "Sigmoid Prob": f"{p.probability:0.4f}",
                        "Raw Logit": f"{p.raw_logit:0.4f}",
                        "Operating Threshold": f"{p.operating_threshold:0.4f}",
                    }
                )

            table_df = pd.DataFrame(table_rows)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

        # Technical Metadata Expander
        with st.expander("🛠️ Technical Execution & Provenance Metadata", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown(f"**Analysis ID:** `{current_result.analysis_id}`")
                st.markdown(f"**Timestamp (UTC):** `{current_result.timestamp}`")
                st.markdown(f"**Model Identifier:** `{current_result.model_name}`")
                st.markdown(f"**Weights:** `{current_result.model_weights}`")
            with t_col2:
                st.markdown(f"**Execution Device:** `{current_result.device}`")
                st.markdown(f"**Inference Latency:** `{format_duration_ms(current_result.inference_time_ms)}`")
                st.markdown(f"**Image Dimensions:** `{current_result.image_metadata.width}×{current_result.image_metadata.height}`")
                st.markdown(f"**Source File:** `{current_result.image_metadata.filename}`")


if __name__ == "__main__":
    main()
