
# ==========================================
# NEUROSCAN AI - STREAMLIT DEMO
# ==========================================

import sys
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.inference import NeuroScanPredictor
from src.preprocessing import eval_transform

DEMO_DIR = BASE_DIR / "data/demo_samples"
MANIFEST_PATH = DEMO_DIR / "demo_manifest.csv"

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}

CLASSES = list(DISPLAY_NAMES.keys())

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)


# ==========================================
# 2. LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_predictor():

    # CPU-only inference for the prototype.
    # The app never attempts to use either HPC GPU.
    predictor = NeuroScanPredictor(
        project_dir=BASE_DIR,
        device="cpu"
    )

    # Protect shared model hooks during Grad-CAM
    inference_lock = threading.Lock()

    return predictor, inference_lock


@st.cache_data
def load_gallery():

    return pd.read_csv(MANIFEST_PATH)


gallery = load_gallery()


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("🧠 NeuroScan AI")


st.markdown(
    """
    **Explainable Brain MRI Classification**

    NeuroScan AI is an educational deep learning prototype
    that classifies brain MRI images into four categories:

    **Glioma | Meningioma | No Tumor | Pituitary**

    The application uses a fine-tuned ResNet18 V2,
    temperature-scaled probabilities, and Grad-CAM
    explanations to demonstrate model predictions.
    """
)


st.warning(
    "Educational research prototype only. "
    "Not validated for medical diagnosis, "
    "clinical screening, or patient care."
)

st.divider()


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("About NeuroScan AI")

    st.markdown(
        """
        **Architecture:** ResNet18

        **Model:** Fine-Tuned ResNet18 V2

        **Starting Checkpoint:** Optuna Trial 1

        **Selected V2 Epoch:** 4

        **Calibration:** Temperature Scaling (T = 1.5373)

        **Explainability:** Grad-CAM

        **Review Threshold:** 97% (Experimental)
        """
    )

    st.divider()

    st.subheader("Upload MRI")

    st.file_uploader(
        "Upload your own MRI",
        type=["png", "jpg", "jpeg"],
        disabled=True,
        help=(
            "Image upload is disabled in this "
            "public demonstration."
        )
    )

    st.caption(
        "For this prototype, please select "
        "one of the preloaded MRI examples."
    )

    st.divider()

    st.caption(
        "Model probabilities and Grad-CAM "
        "are not medically validated measures."
    )


# ==========================================
# 5. IMAGE SELECTION GALLERY
# ==========================================

st.header("1. Select an MRI Image")

st.write(
    "Choose one of the preloaded examples "
    "to explore the model's behavior."
)

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = (
        gallery.iloc[0]["sample_id"]
    )

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

tabs = st.tabs([
    DISPLAY_NAMES[name] for name in CLASSES
])

for tab, class_name in zip(tabs, CLASSES):

    with tab:

        class_gallery = gallery[
            gallery["label"] == class_name
        ]

        columns = st.columns(3)

        for column, (_, row) in zip(
            columns,
            class_gallery.iterrows()
        ):

            sample_id = row["sample_id"]
            image_path = DEMO_DIR / row["filename"]

         
            with column:

                # Display a compact MRI thumbnail
                st.image(
                    str(image_path),
                    width=220
                )

                st.caption(sample_id)

                if st.button(
                    "Select this MRI",
                    key=f"select_{sample_id}",
                    width="stretch"
                ):

                    st.session_state.selected_sample = sample_id

                    # Clear previous prediction when
                    # a different MRI is selected
                    st.session_state.prediction_result = None



# ==========================================
# 6. SELECTED IMAGE
# ==========================================

selected_id = st.session_state.selected_sample

selected_rows = gallery[
    gallery["sample_id"] == selected_id
]

if selected_rows.empty:
    st.error("Selected MRI was not found.")
    st.stop()

selected_row = selected_rows.iloc[0]

selected_path = (
    DEMO_DIR / selected_row["filename"]
).resolve()

# Ensure the selected file is in our demo directory
if not selected_path.is_relative_to(DEMO_DIR.resolve()):
    st.error("Invalid demo image path.")
    st.stop()

st.divider()

st.header("2. Analyze Selected MRI")

left, right = st.columns([1, 2])

with left:

    st.image(
        str(selected_path),
        caption=f"Selected: {selected_id}",
        use_container_width=True
    )

with right:

    st.write("**Selected example:**", selected_id)

    st.write(
        "**Dataset label:**",
        DISPLAY_NAMES[selected_row["label"]]
    )

    st.write(
        "Click the button below to run "
        "the classification model."
    )

    analyze = st.button(
        "Analyze MRI",
        type="primary",
        use_container_width=True
    )


# ==========================================
# 7. RUN MODEL INFERENCE
# ==========================================

if analyze:

    with st.spinner("Analyzing MRI..."):

        predictor, inference_lock = load_predictor()

        with inference_lock:

            prediction = predictor.predict(
                image_path=selected_path,
                explain=True
            )

        st.session_state.prediction_result = (
            prediction
        )


# ==========================================
# 8. PREDICTION RESULTS
# ==========================================

result = st.session_state.prediction_result

if result is not None:

    st.divider()
    st.header("3. AI Prediction Results")

    predicted_class = result["predicted_class"]
    probability = result["max_probability"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Class",
            DISPLAY_NAMES[predicted_class]
        )

    with col2:
        st.metric(
            "Maximum Model Probability",
            f"{probability:.2%}"
        )

    with col3:
        st.metric(
            "Review Threshold",
            f"{result['review_threshold']:.0%}"
        )

    # ======================================
    # 9. REVIEW FLAG
    # ======================================

   
    # ======================================
    # 9. REVIEW FLAG
    # ======================================

    if result["review_recommended"]:

        st.warning(
            "Review recommended: The model's "
            "maximum calibrated probability is below "
            f"the experimental {result['review_threshold']:.0%} "
            "threshold. This does not determine "
            "whether the prediction is correct."
        )

    else:

        st.info(
            "No low-confidence warning triggered. "
            "This does not mean the prediction is "
            "correct or medically reliable. "
            "High-confidence errors are possible."
        )


    # ======================================
    # 10. CLASS PROBABILITIES
    # ======================================

    st.subheader("Class Probabilities")

    for class_name, prob in result[
        "probabilities"
    ].items():

        label_col, value_col = st.columns([4, 1])

        with label_col:
            st.write(DISPLAY_NAMES[class_name])

        with value_col:
            st.write(f"{prob:.2%}")

        st.progress(float(prob))

    st.caption(
        "These are temperature-scaled model "
        "probabilities, not clinical diagnostic "
        "probabilities."
    )

    # ======================================
    # 11. GRAD-CAM VISUALIZATION
    # ======================================

    st.subheader("Grad-CAM Explanation")

    with Image.open(selected_path) as img:
        original_image = img.convert("RGB").copy()

    # Recreate the exact model input for alignment
    image_tensor = eval_transform(original_image)

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    display_tensor = (
        image_tensor * std + mean
    ).clamp(0, 1)

    display_array = (
        display_tensor
        .permute(1, 2, 0)
        .numpy()
    )

    base_image = Image.fromarray(
        (display_array * 255).astype(np.uint8)
    )

    heatmap = result["heatmap"]

    colored_heatmap = (
        plt.get_cmap("jet")(heatmap)[..., :3] * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        colored_heatmap
    )

    overlay_image = Image.blend(
        base_image,
        heatmap_image,
        alpha=0.40
    )

    img_col1, img_col2 = st.columns(2)

    with img_col1:

        st.image(
            base_image,
            caption="Preprocessed MRI",
            use_container_width=True
        )

    with img_col2:

        st.image(
            overlay_image,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )

    st.caption(
        "Grad-CAM highlights image regions "
        "contributing to the predicted class "
        "score. It does not identify or "
        "segment a tumor."
    )

    
    # ======================================
    # 12. MODEL INFORMATION
    # ======================================

    with st.expander("Model information"):

        st.write("Architecture: ResNet18")

        st.write(
            "Model version:",
            result["model_stage"]
        )

        st.write(
            "Best checkpoint epoch:",
            result["model_epoch"]
        )

        st.write(
            "Calibration temperature:",
            round(result["temperature"], 6)
        )

        st.write(
            "Experimental review threshold:",
            f"{result['review_threshold']:.0%}"
        )

        st.divider()

        st.write(
            "**Development Validation Results**"
        )

        st.write(
            "Accuracy: 97.61%"
        )

        st.write(
            "Macro F1-score: 97.69%"
        )

        st.write(
            "Correct predictions: 694 / 711"
        )

        st.caption(
            "Metrics were measured on development "
            "validation data used for model selection. "
            "They are not independent test results "
            "or clinically validated performance estimates."
        )

        st.warning(
            "The confidence-based review mechanism "
            "can miss incorrect predictions, including "
            "high-confidence tumor-to-no-tumor errors."
        )

