import streamlit as st
from PIL import Image

from infer import predict_image
from sklearn.metrics import roc_auc_score

import hashlib

st.title("MVTec Anomaly Detection")

category = st.selectbox(
    "Select product category",
    ["hazelnut", "wood"]
)

threshold = st.slider(
    "Threshold",
    min_value=3.0,
    max_value=7.0,
    value=4.7,
    step=0.1
)

# it helps to set ground truth value before uploading image batch. it helps to reduce hand work
default_ground_truth = st.selectbox(
    "Default Ground Truth Label",
    ["NORMAL", "ANOMALY"]
)

uploaded_files = st.file_uploader(
    "Choose images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:

    # true positive, false positive, etc..
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # for AUROC evaluation
    all_scores = []
    all_labels = []

    st.write(f"Selected category: **{category}**")
    st.write(f"{len(uploaded_files)} image(s) uploaded.")

    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")

        result = predict_image(
            image=image,
            category=category,
            threshold=threshold
        )

        image_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(image_bytes).hexdigest()

        ground_truth = st.radio(
            f"Ground Truth for {uploaded_file.name}",
            ["NORMAL", "ANOMALY"],
            index=0 if default_ground_truth == "NORMAL" else 1,
            key=f"ground_truth_{file_hash}"
        )

        prediction = result["prediction"]

        if ground_truth == "NORMAL" and prediction == "NORMAL":
            tn += 1

        elif ground_truth == "NORMAL" and prediction == "ANOMALY":
            fp += 1

        elif ground_truth == "ANOMALY" and prediction == "ANOMALY":
            tp += 1

        elif ground_truth == "ANOMALY" and prediction == "NORMAL":
            fn += 1

        label = 1 if ground_truth == "ANOMALY" else 0

        all_labels.append(label)
        all_scores.append(result["anomaly_score"])

        st.subheader(uploaded_file.name)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with col2:
            st.image(
                result["overlay"],
                caption="Anomaly Heatmap Overlay",
                use_container_width=True
            )

        st.write(f"Prediction: **{result['prediction']}**")
        st.write(f"Anomaly Score: `{result['anomaly_score']:.2f}`")

        if result["prediction"] == "ANOMALY":
            st.error("Anomaly Detected")
        else:
            st.success("Normal Image")

    st.divider()

    st.subheader("Evaluation Summary")

    st.write(f"True Positive (TP): {tp}")
    st.write(f"True Negative (TN): {tn}")
    st.write(f"False Positive (FP): {fp}")
    st.write(f"False Negative (FN): {fn}")

    # Total samples
    total = tp + tn + fp + fn

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    st.divider()

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1_score:.4f}")

    # AUROC
    try:
        auroc = roc_auc_score(all_labels, all_scores)
        st.write(f"AUROC: {auroc:.4f}")
    except:
        st.write("AUROC could not be calculated.")