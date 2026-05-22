import streamlit as st
from PIL import Image

from infer import predict_image

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

    st.write(f"Selected category: **{category}**")
    st.write(f"{len(uploaded_files)} image(s) uploaded.")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        result = predict_image(
            image=image,
            category=category,
            threshold=threshold
        )

        ground_truth = st.radio(
            f"Ground Truth for {uploaded_file.name}",
            ["NORMAL", "ANOMALY"],
            key=uploaded_file.name
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

        st.subheader(uploaded_file.name)

        st.image(
            image,
            caption="Uploaded Image",
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