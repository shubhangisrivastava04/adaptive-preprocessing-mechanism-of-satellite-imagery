import numpy as np
import rasterio
from skimage import exposure
from skimage.transform import resize
from scipy.ndimage import minimum_filter, gaussian_filter
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

def predict_segmentation_with_patches(full_img, model, patch_size=64, stride=64, batch_size=16):
    H, W, C = full_img.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    img_padded = np.pad(full_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    # Extract patches
    patches = view_as_windows(img_padded, (patch_size, patch_size, C), step=stride)
    patches = patches.reshape(-1, patch_size, patch_size, C)

    # Predict
    pred_probs = model.predict(patches, batch_size=batch_size)
    pred_labels = pred_probs.argmax(axis=-1)  # shape: (N, 64, 64)

    # Stitch back
    H_pad, W_pad = img_padded.shape[:2]
    pred_map = np.zeros((H_pad, W_pad), dtype=np.uint8)

    index = 0
    for i in range(0, H_pad, patch_size):
        for j in range(0, W_pad, patch_size):
            pred_map[i:i+patch_size, j:j+patch_size] = pred_labels[index]
            index += 1

    # Remove padding
    return pred_map[:H, :W]

# ===== Preprocessing Functions =====

def normalize(bands):
    return np.clip(bands / 10000.0, 0, 1)

def histogram_stretch(bands):
    stretched = np.zeros_like(bands)
    for i in range(bands.shape[0]):
        p2, p98 = np.percentile(bands[i], (2, 98))
        stretched[i] = exposure.rescale_intensity(bands[i], in_range=(p2, p98))
    return stretched

def apply_clahe(bands):
    bands = np.clip(bands / 10000.0, 0, 1)
    clahe_bands = np.zeros_like(bands)
    for i in range(bands.shape[0]):
        clahe_bands[i] = exposure.equalize_adapthist(bands[i], clip_limit=0.03)
    return clahe_bands

def apply_cloud_mask(bands):
    red = bands[3]    # B4
    nir = bands[7]    # B8
    ndvi = (nir - red) / (nir + red + 1e-5)
    brightness = red
    cloud_mask = (ndvi < 0.1) & (brightness > 0.25)
    bands[:, cloud_mask] = 0
    return bands

def compute_ndvi(bands):
    red = bands[3]    # B4
    nir = bands[7]    # B8
    return (nir - red) / (nir + red + 1e-5)

# ===== Thin Cloud Removal (RGB) =====

def dark_channel(image, size=15):
    min_channel = np.min(image, axis=0)
    return minimum_filter(min_channel, size=size)

def linear_transform(source, target):
    a, b = np.polyfit(source.flatten(), target.flatten(), 1)
    return a * source + b, a, b

def estimate_local_atmospheric_light(image, patch_size=100):
    A = np.zeros_like(image)
    h, w = image.shape[1:]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            dark = np.min(patch, axis=0)
            idx = np.unravel_index(np.argmax(dark), dark.shape)
            for c in range(3):
                A[c, y:y+patch_size, x:x+patch_size] = patch[c, idx[0], idx[1]]
    return gaussian_filter(A, sigma=(0, patch_size // 2, patch_size // 2))

def estimate_transmission(dark_map, A_band, a=1.0, b=0.0):
    return 1.0 - ((dark_map - b) / (a * A_band + 1e-6))

def recover_image(I, t, A):
    J = np.zeros_like(I)
    for c in range(3):
        J[c] = (I[c] - A[c]) / np.maximum(t[c], 0.1) + A[c]
    return np.clip(J, 0, 10000).astype(np.uint16)

def remove_thin_clouds_rgb(bands):
    red   = bands[3]  # B4
    green = bands[2]  # B3
    blue  = bands[1]  # B2
    I = np.stack([red, green, blue])
    A = estimate_local_atmospheric_light(I)
    dark_r = dark_channel(I)
    t_r = estimate_transmission(dark_r, A[0])
    green_to_red, a_gr, b_gr = linear_transform(green, red)
    t_g = estimate_transmission(dark_channel(np.stack([green_to_red, red, blue])), A[1], a_gr, b_gr)
    blue_to_red, a_br, b_br = linear_transform(blue, red)
    t_b = estimate_transmission(dark_channel(np.stack([blue_to_red, green, red])), A[2], a_br, b_br)
    t = np.stack([t_r, t_g, t_b])
    result_rgb = recover_image(I, t, A) / 10000.0
    bands[3], bands[2], bands[1] = result_rgb  # Put back to B4, B3, B2
    return bands

# ===== Adaptive Preprocessing =====

def adaptive_preprocessing_from_model(image_path, model, threshold=0.5):
    with rasterio.open(image_path) as src:
        bands = src.read().astype(np.float32)  # shape: (12, H, W)

    # Preparing input RGB for prediction (B4, B3, B2 → indices 3,2,1)
    rgb = np.stack([bands[3], bands[2], bands[1]], axis=-1) / 10000.0
    rgb_resized = resize(rgb, (128, 128), anti_aliasing=True)
    pred = model.predict(np.expand_dims(rgb_resized, axis=0))[0]
    pred_bin = (pred > threshold).astype(int)

    if pred_bin[0]: bands = remove_thin_clouds_rgb(bands)
    if pred_bin[1]: bands = apply_cloud_mask(bands)
    if pred_bin[4]: bands = normalize(bands)
    if pred_bin[2]: bands = histogram_stretch(bands)
    if pred_bin[3]: bands = apply_clahe(bands)

    return bands, pred_bin

def get_rgb(bands):
    return np.stack([bands[3], bands[2], bands[1]], axis=-1)  # B4, B3, B2 (R, G, B)

def extract_rgb_adaptive(img_path, steps_applied):
    s2_img = rasterio.open(img_path)
    s2_data = s2_img.read()  

    # === Extract RGB ===
    rgb = np.stack([
        s2_data[3],  # B4 - Red
        s2_data[2],  # B3 - Green
        s2_data[1]   # B2 - Blue
    ], axis=-1)

    # === Normalize for display ===
    rgb = rgb.astype(np.float32)
    rgb /= np.percentile(rgb, 98)  
    rgb = np.clip(rgb, 0, 1)

    # === Preprocessing vector and labels ===
    steps_applied = steps_applied 
    step_names = ["Thin Cloud Removal", "Cloud Mask", "Histogram Stretching", "CLAHE", "Normalization"]
    applied_steps = [step_names[i] for i, val in enumerate(steps_applied) if val == 1]

    # === Text formatting ===
    vector_str = f"Vector: {steps_applied}"
    steps_str = "Steps Applied:\n" + "\n".join(f"- {s}" for s in applied_steps) if applied_steps else "None"

    # === Plot and annotate ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(rgb)
    ax.set_title("Preprocessed Sentinel-2 RGB (B4, B3, B2)", fontsize=14)
    ax.axis('off')

    fig.text(0.5, -0.07, steps_str, ha='center', fontsize=12, color='black')
    return rgb

def plot_full_landcover(pred_map, save_path=None):
    """
    Plots full land cover map with all 6 classes. Optionally saves to a file.
    """
    custom_palette = [
        '#419bdf',  # Water
        '#397d49',  # Trees
        '#88b053',  # Grass/Shrub
        '#e49635',  # Crops/Flooded Vegetation
        '#c4281b',  # Built-up Area
        '#a59b8f',  # Bare Land
    ]
    class_labels = [
        'Water',
        'Trees',
        'Grass/Shrub',
        'Crops/Flooded Vegetation',
        'Built-up Area',
        'Bare Land'
    ]

    cmap = ListedColormap(custom_palette)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, 6.5, 1), ncolors=6)

    plt.figure(figsize=(8, 6))
    plt.imshow(pred_map, cmap=cmap, norm=norm)
    plt.title("Predicted Land Cover")
    plt.axis("off")

    legend_patches = [mpatches.Patch(color=custom_palette[i], label=class_labels[i]) for i in range(6)]
    plt.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_vegetation_only(pred_map, save_path=None):
    """
    Plots vegetation-focused map with Trees, Grass/Shrub, and Crops/Flooded highlighted.
    Others shown in gray. Optionally saves to a file.
    """
    full_palette = [
        '#808080',  # Water → gray
        '#397d49',  # Trees (highlight)
        '#88b053',  # Grass/Shrub (highlight)
        '#e49635',  # Crops/Flooded (highlight)
        '#808080',  # Built-up → gray
        '#808080',  # Bare → gray
    ]
    cmap = ListedColormap(full_palette)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, 6.5, 1), ncolors=6)

    legend_labels = {
        'Trees': '#397d49',
        'Grass/Shrub': '#88b053',
        'Crops/Flooded': '#e49635',
        'Others': '#808080'
    }

    plt.figure(figsize=(8, 6))
    plt.imshow(pred_map, cmap=cmap, norm=norm)
    plt.title("Land Cover (Only Key Classes Highlighted)")
    plt.axis("off")

    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in legend_labels.items()]
    plt.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()    