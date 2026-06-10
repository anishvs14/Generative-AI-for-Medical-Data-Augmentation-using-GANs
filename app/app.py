import os
from PIL import Image

import pandas as pd
import streamlit as st

from model_utils import (
    MODEL_PATHS,
    predict_image,
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# PATHS
# =========================
BASE_DIR = "/content/drive/MyDrive/generative_ai_project"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

def first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

ROC_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")
PR_PATH = os.path.join(RESULTS_DIR, "pr_curve.png")
CM_PATH = os.path.join(RESULTS_DIR, "confusion_test_mapped.png")

GAN_GALLERY_PATH = first_existing_path([
    os.path.join(OUTPUT_DIR, "fake_samples_epoch_110.png"),
    os.path.join(OUTPUT_DIR, "fake_samples_epoch_200.png"),
])

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e5e7eb;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc !important;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        font-size: 1rem;
        color: #cbd5e1 !important;
        margin-bottom: 1rem;
    }

    .card {
        background: #111827;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.15);
        color: #e5e7eb !important;
        min-height: 95px;
    }

    .card h4, .card p, .card b, .card div, .card span {
        color: #e5e7eb !important;
        margin: 0;
    }

    .small-label {
        font-size: 0.85rem;
        color: #94a3b8 !important;
        margin-bottom: 0.25rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    .result-box {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        font-size: 1.15rem;
        font-weight: 800;
        text-align: center;
        border: 1px solid #334155;
        background: #111827;
        color: #f8fafc !important;
    }

    .normal-box {
        background: #052e16;
        color: #bbf7d0 !important;
        border: 1px solid #166534;
    }

    .pneumonia-box {
        background: #450a0a;
        color: #fecaca !important;
        border: 1px solid #991b1b;
    }

    .info-note {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #e5e7eb;
    }

    .soft-divider {
        border-top: 1px solid #1f2937;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩻 Project Info")
st.sidebar.markdown(
    """
**Project:** Generative AI for Medical Data Augmentation  
**Task:** Pneumonia detection from chest X-rays  
**Classifier:** DenseNet121  
**Environment:** Google Colab + Streamlit  
**Storage:** Google Drive
"""
)

selected_model = st.sidebar.selectbox(
    "Choose prediction model",
    options=list(MODEL_PATHS.keys()),
    index=1,  # default: Traditional Augmentation (B)
)

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.01,
)

st.sidebar.markdown("### How to use")
st.sidebar.markdown(
    """
1. Upload a chest X-ray image  
2. Select a model  
3. Click **Predict**  
4. View the result and confidence  
5. Explore the GAN training story and performance plots
"""
)

st.sidebar.warning(
    "This is a research prototype and is not a clinical diagnostic tool."
)

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">🩺 AI Pneumonia Detection System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="sub-title">
    Upload a chest X-ray image, choose a trained classifier, and get a real-time prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1.5, 1, 1.2])
with c1:
    st.markdown(
        """
        <div class="card">
            <div class="small-label">Pipeline</div>
            <b>GAN augmentation → filtered synthetic data → DenseNet121 classifier → web UI</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="card">
            <div class="small-label">Selected model</div>
            <b>{selected_model}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="card">
            <div class="small-label">Threshold</div>
            <b>{threshold:.2f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Predict", "📊 Performance", "🧠 GAN Training Story", "ℹ️ About Project"]
)

# =========================
# TAB 1: PREDICT
# =========================
with tab1:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("### Upload Chest X-ray")
        uploaded_file = st.file_uploader(
            "Choose a PNG, JPG or JPEG image",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

            predict_clicked = st.button("Predict", use_container_width=True)

            if predict_clicked:
                with st.spinner("Analyzing image..."):
                    try:
                        result = predict_image(image, selected_model, threshold=threshold)
                        st.session_state["prediction_result"] = result
                    except Exception as e:
                        st.session_state["prediction_result"] = None
                        st.error(f"Prediction failed: {e}")

    with right:
        st.markdown("### Result")
        result = st.session_state.get("prediction_result", None)

        if result:
            label = result["label"]
            confidence = result["confidence"]
            probability = result["probability"]

            if label == "PNEUMONIA":
                st.markdown(
                    f'<div class="result-box pneumonia-box">Prediction: {label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="result-box normal-box">Prediction: {label}</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.progress(float(confidence))
            st.caption(f"Raw model probability: {probability:.4f}")
        else:
            st.markdown(
                """
                <div class="info-note">
                    Upload an image and press <b>Predict</b> to see the result here.
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================
# TAB 2: PERFORMANCE
# =========================
with tab2:
    st.markdown("### Final Model Comparison")

    perf_df = pd.DataFrame(
        [
            ["Baseline (A)", 0.82, 0.53, 1.00, 0.78],
            ["Traditional Augmentation (B)", 0.88, 0.71, 0.99, 0.87],
            ["GAN Augmentation (C)", 0.87, 0.65, 0.99, 0.84],
            ["GAN + Traditional (D)", 0.88, 0.68, 0.99, 0.86],
        ],
        columns=["Model", "Accuracy", "NORMAL Recall", "PNEUMONIA Recall", "Macro F1"],
    )
    st.table(perf_df)

    st.markdown("### Evaluation Plots")
    p1, p2 = st.columns(2)

    with p1:
        if os.path.exists(ROC_PATH):
            st.image(ROC_PATH, caption="ROC Curve", use_container_width=True)
        else:
            st.warning("ROC curve not found.")

    with p2:
        if os.path.exists(PR_PATH):
            st.image(PR_PATH, caption="Precision-Recall Curve", use_container_width=True)
        else:
            st.warning("Precision-Recall curve not found.")

    st.write("")
    if os.path.exists(CM_PATH):
        st.markdown("### Confusion Matrix")
        st.image(CM_PATH, caption="Confusion Matrix", use_container_width=True)

    st.markdown(
        """
        **Why these plots matter**
        - ROC shows class separation across thresholds.
        - PR curve is important for imbalanced medical data.
        - Confusion matrix shows correct and incorrect predictions.
        """
    )

# =========================
# TAB 3: GAN TRAINING STORY
# =========================
with tab3:
    st.markdown("### The Story of Our GAN Training")
    st.caption(
        "WGAN-GP was trained for 110 epochs to generate realistic NORMAL chest X-ray images."
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Training Epochs</div>
                <b>110</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">GAN Type</div>
                <b>WGAN-GP</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="card">
                <div class="small-label">Purpose</div>
                <b>Generate NORMAL chest X-rays</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### GAN-Generated Image Gallery")

    if os.path.exists(GAN_GALLERY_PATH):
        st.image(
            GAN_GALLERY_PATH,
            caption="GAN-generated chest X-ray samples",
            use_container_width=True,
        )
    else:
        st.warning("GAN sample grid not found yet.")

    st.write("")
    st.info(
        "The GAN story tab shows the synthetic NORMAL images generated by the best checkpoint. "
        "These samples were used to support the augmentation pipeline."
    )

# =========================
# TAB 4: ABOUT PROJECT
# =========================
with tab4:
    st.markdown("### Project Overview")

    a, b, c = st.columns(3)
    with a:
        st.markdown(
            """
            <div class="card">
                <h4>1. GAN Training</h4>
                <p>The WGAN-GP learns to generate realistic NORMAL chest X-ray images.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="card">
                <h4>2. Filtering</h4>
                <p>Generated images are filtered using DenseNet121 feature similarity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="card">
                <h4>3. Classification</h4>
                <p>The final DenseNet121 classifier is trained using real and synthetic data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### Final Experiment Setup")
    st.markdown(
        """
        - **Model A:** No augmentation  
        - **Model B:** Traditional augmentation only  
        - **Model C:** GAN augmentation only  
        - **Model D:** GAN + Traditional augmentation  
        """
    )

    st.write("")
    st.info(
        "This project demonstrates an end-to-end Generative AI pipeline for medical image augmentation and pneumonia detection."
    )

    st.write("")
    st.markdown("### Model Selection")
    st.markdown(
        f"""
        Current selected model for prediction: **{selected_model}**
        """
    )
