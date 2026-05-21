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
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.5
)

uploaded_files = st.file_uploader(
    "Choose images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Selected category: **{category}**")
    st.write(f"{len(uploaded_files)} image(s) uploaded.")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        result = predict_image(
            image=image,
            category=category,
            threshold=threshold
        )

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