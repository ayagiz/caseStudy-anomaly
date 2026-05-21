import streamlit as st
from PIL import Image

from infer import predict_image

# Başlık
st.title("MVTec Anomaly Detection")

st.write("Upload one or more images.")

# Threshold slider
threshold = st.slider(
    "Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# Çoklu image upload
uploaded_files = st.file_uploader(
    "Choose images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Eğer image varsa
if uploaded_files:

    st.write(f"{len(uploaded_files)} image(s) uploaded.")

    for uploaded_file in uploaded_files:

        # Image aç
        image = Image.open(uploaded_file)

        # Prediction al
        result = predict_image(image, threshold)

        # Dosya adı
        st.subheader(uploaded_file.name)

        # Görsel göster
        st.image(
            image,
            use_container_width=True
        )

        # Sonuç göster
        st.write(f"Prediction: {result['prediction']}")
        st.write(f"Anomaly Score: {result['anomaly_score']:.2f}")

        # Renkli durum mesajı
        if result["prediction"] == "ANOMALY":
            st.error("Anomaly Detected")
        else:
            st.success("Normal Image")