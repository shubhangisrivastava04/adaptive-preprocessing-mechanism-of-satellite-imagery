# 🌿 Adaptive Preprocessing Mechanism of Satellite Imagery for Vegetation Land Cover Identification

This project presents an **adaptive preprocessing framework** for enhancing the quality of **Sentinel-2 satellite imagery** to improve **vegetation land cover segmentation**.  
Traditional preprocessing applies the same enhancement techniques to all images — often leading to over- or under-processing.  
Our system uses a **shallow multi-label CNN** to predict which preprocessing steps are required for each image patch and a **U-Net++ model** to perform pixel-wise vegetation classification.

---

## 🧠 Overview

The project integrates **two deep learning components**:
1. **Adaptive Preprocessing Prediction**  
   A shallow multi-label CNN predicts which preprocessing steps (cloud masking, normalization, CLAHE, etc.) should be applied for each image patch.
2. **Land Cover Segmentation**  
   A U-Net++ segmentation model classifies each pixel into one of six land cover classes:
   - 🌳 Trees  
   - 🌾 Crops & Flooded Vegetation  
   - 🌿 Shrubs & Grass  
   - 💧 Waterbodies  
   - 🏙 Built-up Area  
   - 🏜 Bare Land

Together, these modules form an **intelligent preprocessing + segmentation pipeline** that adapts to image conditions, reduces redundancy, and enhances classification accuracy.

---

## 🧩 Project Structure
```
adaptive-preprocessing-mechanism-of-satellite-imagery/

│

├── app.py # Streamlit app for demo visualization

├── requirements.txt # List of dependencies to install

│

├── processing/

│ ├── preprocessing_utils.py # Functions for cloud masking, normalization, CLAHE, etc.

│ ├── pipeline.py # Pipeline that neatly integrates adaptie preprocessing with vegetation segmentation

│ ├── adaptive_model.keras # Trained shallow CNN for preprocessing prediction

│ ├── segmentation_model.keras # Trained U-Net++ segmentation model

│

├── test_imgs/ # Example Sentinel-2 input images in .TIF Format

│

└── README.md # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/shubhangisrivastava04/adaptive-preprocessing-mechanism-of-satellite-imagery.git
cd adaptive-preprocessing-mechanism-of-satellite-imagery
```

### 2. Install Git LFS (for model files)
If you haven't already:
```
git lfs install
git lfs pull
```

### 3. Set Up a Virtual Environment (Optional but Recommended)
```
python -m venv venv
# Activate the virtual environment:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Install Dependencies
Make sure you have Python 3.8+ installed.
Then install all required packages using:
```
pip install -r requirements.txt
```

### 5. Run the Streamlit App
Once dependencies are installed, launch the interactive demo:
```
streamlit run app.py
```

The Streamlit GUI will allow you to:

1. Upload or select a sample Sentinel-2 image

2. View preprocessing predictions made by the CNN

3. Generate pixel-wise vegetation segmentation using U-Net++

---

## 📁 Folder Details

### `processing/`
Contains:

#### 🧠 Trained Models
- `adaptive_model.keras` – predicts required preprocessing steps  
- `segmentation_model.keras` – performs segmentation  

#### ⚙️ Preprocessing Utilities
Implementation of:
- Cloud Masking  
- CLAHE  
- Normalization  
- Histogram Stretching  
- Thin Cloud Removal  

#### 🔄 Adaptive Pipeline
Dynamically applies the predicted preprocessing sequence before segmentation.

---

## 🧪 Key Results

| Model | Accuracy | Mean IoU | Highlights |
|--------|-----------|----------|-------------|
| **U-Net++** | 92.12% | 81.39% | Highest segmentation accuracy |
| **CNN Preprocessor** | F1 = 0.75 | — | Efficient and selective preprocessing |
| **Best Classes** | Water (IoU 93.47%), Trees (IoU 88.92%) |  |  |

> Adaptive preprocessing demonstrated improved segmentation accuracy compared to static pipelines, especially under clouded or low-contrast conditions.

---

## 🏗️ Tech Stack

- **Language:** Python  
- **Frameworks:** PyTorch, Streamlit  
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn  
- **Dataset:** Sentinel-2 imagery via Google Earth Engine  
- **Ground Truth:** Dynamic World dataset (Google Earth Engine)

---

## 👩‍💻 Contributors

| Name | Role |
|------|------|
| **Shubhangi Srivastava** | Research Student |
| **Sneha Shetty** | Research Student |
| **Prof. Pavithra S** | Project Mentor |

> Conducted under **Centre of Data Modelling, Analytics and Visualization (CoDMAV)**,  
> **Department of Computer Science and Engineering, PES University**, Bengaluru, India.

---

## 🚀 Future Work

- Incorporating **spectral unmixing** and **shadow removal** into preprocessing.  
- Expanding datasets to cover **more geographies and seasons**.  
- Introducing **temporal analysis** for vegetation change monitoring.  
- Optimizing model inference for **real-time environmental monitoring**.
