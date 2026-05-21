import streamlit as st
from PIL import Image

# Sayfa başlığı
st.title("MVTec Anomaly Detection")

# Açıklama
st.write("Upload one or more images.")

# Çoklu image upload
uploaded_files = st.file_uploader(
    "Choose images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# Eğer image yüklenirse
if uploaded_files:

    st.write(f"{len(uploaded_files)} image(s) uploaded.")

    # Tüm image'leri göster
    for uploaded_file in uploaded_files:

        # Image aç
        image = Image.open(uploaded_file)

        # Dosya adını göster
        st.subheader(uploaded_file.name)

        # Ön izleme
        st.image(
            image,
            use_container_width=True
        )

    st.success("Images uploaded successfully!")