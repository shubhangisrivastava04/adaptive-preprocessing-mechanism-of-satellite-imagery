import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import tensorflow as tf

from processing.preprocessing_utils import (
    adaptive_preprocessing_from_model,
    compute_ndvi,
    get_rgb,
    extract_rgb_adaptive,
    predict_segmentation_with_patches,
    plot_full_landcover,
    plot_vegetation_only
)

# === Custom loss function needed for adaptive model loading ===
def weighted_loss(y_true, y_pred):
    w = tf.constant([1.0, 3.5, 1.0, 4.0, 2.0])
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * w)

# === Load models ===
adaptive_model = tf.keras.models.load_model(
    "processing/adaptive_model.keras",
    custom_objects={'weighted_loss': weighted_loss}
)

segmentation_model = tf.keras.models.load_model(
    "processing/segmentation_model.keras"
)

# === Full image processing pipeline ===
def run_full_pipeline(input_tif_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    s2_img = rasterio.open(input_tif_path)
    s2_data = s2_img.read()

    rgb = np.stack([
        s2_data[3],
        s2_data[2],
        s2_data[1]
    ], axis = -1)

    rgb = rgb.astype(np.float32)
    rgb /= np.percentile(rgb, 98)

    # === Save uploaded image as PNG ===
    uploaded_png_path = os.path.join(output_dir, "uploaded.png")
    plt.figure(figsize = (8,8))
    plt.imshow(np.clip(rgb, 0, 1))
    plt.title("Sentinel-2 RGB (Uploaded Image)")
    plt.axis("off")
    plt.savefig(uploaded_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    # === Adaptive preprocessing using model ===
    processed_bands, steps_applied = adaptive_preprocessing_from_model(input_tif_path, adaptive_model)

    # === Save preprocessed TIF ===
    preprocessed_tif_path = os.path.join(output_dir, "preprocessed.tif")
    with rasterio.open(input_tif_path) as src:
        meta = src.meta.copy()

    meta.update({
        "dtype": rasterio.float32,
        "count": processed_bands.shape[0]
    })

    with rasterio.open(preprocessed_tif_path, 'w', **meta) as dst:
        dst.write(processed_bands.astype(np.float32))

    # === Save preprocessed PNG ===
    preprocessed_png_path = os.path.join(output_dir, "preprocessed.png")
    preprocessed_rgb = extract_rgb_adaptive(preprocessed_tif_path, steps_applied)
    plt.imshow(preprocessed_rgb)
    plt.axis("off")
    plt.savefig(preprocessed_png_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Preprocess input to expected range (0-1)
    with rasterio.open(preprocessed_tif_path) as src:
        full_img = src.read().astype(np.float32).transpose(1,2,0)

    full_img /= np.max(full_img)

    seg_full = predict_segmentation_with_patches(full_img, segmentation_model)

    # === Save full segmentation map ===
    seg_full_path = os.path.join(output_dir, "segmented_full.png")
    plot_full_landcover(seg_full, save_path=seg_full_path)

    # === Highlighted vegetation map ===
    seg_highlighted_path = os.path.join(output_dir, "segmented_highlighted.png")
    plot_vegetation_only(seg_full, save_path=seg_highlighted_path)

    # === Return paths to frontend ===
    return {
        "uploaded_png": uploaded_png_path,
        "preprocessed_tif": preprocessed_tif_path,
        "preprocessed_png": preprocessed_png_path,
        "segmented_full": seg_full_path,
        "segmented_highlighted": seg_highlighted_path
    }


