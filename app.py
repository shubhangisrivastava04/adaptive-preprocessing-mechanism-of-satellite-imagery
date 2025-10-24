import streamlit as st
from processing.pipeline import run_full_pipeline
import numpy as np
import os

st.set_page_config(page_title="Vegetation Segmentation Tool", layout="centered")

st.title("🌿 Adaptive Vegetation Segmentation Tool")
st.markdown("Upload a `.tif` satellite image and get segmented vegetation maps and preprocessing outputs.")

# === Upload .tif ===
uploaded_file = st.file_uploader("📤 Upload your `.tif` image", type=["tif", "tiff"])

if uploaded_file:
    # Save the uploaded file to disk
    with open("temp_input.tif", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Image uploaded. Running preprocessing and segmentation...")

    # Full pipeline
    results = run_full_pipeline("temp_input.tif", "outputs")

    st.subheader("📷 Uploaded Image (PNG Preview)")
    st.image(results["uploaded_png"], use_column_width=True)

    st.subheader("🔧 Preprocessed Image")
    st.image(results["preprocessed_png"], caption="With preprocessing steps applied", use_column_width=True)

    st.subheader("🌍 Full Segmentation Map")
    st.image(results["segmented_full"], caption="All land cover classes", use_column_width=True)

    st.subheader("🌿 Highlighted Vegetation Map")
    st.image(results["segmented_highlighted"], caption="Only vegetation classes highlighted", use_column_width=True)

    st.markdown("### 📥 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        with open(results["preprocessed_tif"], "rb") as f:
            st.download_button("Download Preprocessed (TIF)", f, file_name="preprocessed.tif")

        with open(results["preprocessed_png"], "rb") as f:
            st.download_button("Download Preprocessed (PNG)", f, file_name="preprocessed.png")

    with col2:
        with open(results["segmented_full"], "rb") as f:
            st.download_button("Download Full Segmentation", f, file_name="segmented_full.png")

        with open(results["segmented_highlighted"], "rb") as f:
            st.download_button("Download Highlighted Map", f, file_name="segmented_highlighted.png")
