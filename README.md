# Surface Crack Detection, Segmentation & Severity Evaluation

A deep learning-based computer vision pipeline for automated detection, pixel-level segmentation, and geometric severity assessment of structural surface cracks. Built using PyTorch and trained on multi-source Kaggle surface and structural crack datasets.

---

##  Overview

Manual inspection of concrete, masonry, and building surfaces is time-consuming, subjective, and prone to human error. This project provides an end-to-end automated framework capable of:

1. **Binary Classification:** Distinguishing between `Cracked` and `Non-Cracked` structural surfaces using both a custom Convolutional Neural Network (`CrackCNN`) and Transfer Learning (`MobileNetV2`).
2. **Semantic Segmentation:** Generating pixel-level binary masks to isolate crack boundaries from complex background textures.
3. **Severity Quantification:** Computing geometric crack metrics—including **Area (px)**, **Length (px)**, **Average Width (px)**, and **Maximum Width (px)**—alongside visual centerline overlays for structural health monitoring.

---

##  Datasets

The training and evaluation pipeline integrates three public Kaggle datasets via `kagglehub`, supplemented by a custom test set of real-world surface photographs:

| Dataset | Kaggle Identifier | Task | Key Contents |
| :--- | :--- | :--- | :--- |
| **Cracked & Non-Cracked Surface** | `geadalfa/cracked-non-cracked-surface-datasets` | Classification | `Cracked` (28,484 images) & `Non Cracked` (67,608 images) |
| **Crack Segmentation Dataset** | `aravindnagarajan/crack-segmentation-dataset` | Segmentation | `train` (9,603 image/mask pairs) & `test` (1,695 image/mask pairs) |
| **Surface Crack Detection** | `arunrk7/surface-crack-detection` | Classification | `Positive` (20,000 images) & `Negative` (20,000 images) |

---

##  Repository Structure

```text
Crack_Detection_Project/
├── datasets/                          # Dataset documentation and local/Drive setup instructions
│   └── README.md
├── models/                            # Saved PyTorch model checkpoints (.pth) & model notes
│   └── README.md
├── notebooks/                         # Google Colab Jupyter notebook for preprocessing, training & inference
│   ├── Concrete_Crack_Detection.ipynb
│   └── README.md
├── outputs/                           # Generated classification, segmentation masks, and overlay results
│   └── README.md
├── screenshots/                       # UI and sample prediction captures
│   ├── output1.jpeg
│   ├── output2.jpeg
│   ├── output3.jpeg
│   └── README.md
├── .gitignore                         # Ignores large dataset archives and heavy weight files
├── README.md                          # Main project documentation
└── requirements.txt                   # Python dependencies
```

---

##  Tech Stack

* **Deep Learning Framework:** PyTorch, Torchvision
* **Data Handling & Processing:** NumPy, Pillow (PIL), OpenCV
* **Dataset Management:** `kagglehub`
* **Visualization:** Matplotlib
* **Development Environment:** Google Colab (with Google Drive persistence)

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Vaidehigupta05/crack_detection_severity_estimation.git](https://github.com/Vaidehigupta05/crack_detection_severity_estimation.git)
cd crack_detection_severity_estimation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Google Colab Execution
1. Open `notebooks/Concrete_Crack_Detection.ipynb` in Google Colab.
2. Mount your Google Drive to `/content/drive`.
3. Ensure your Kaggle API credentials are configured if required by `kagglehub`.
4. Run the setup, training, and `analyze_concrete_image_safeguarded()` inference cells to initialize `/content/drive/MyDrive/Crack_Detection_Project`, download datasets directly to `/kaggle/input/`, and generate diagnostic outputs.

---

##  Methodology & Pipeline

### 1. Data Preprocessing & Class Balancing
* Images are resized, normalized, and augmented to improve generalization across concrete, plaster, and brick masonry textures.
* To address class imbalance across combined datasets (`Non-Cracked`: 47,325 vs. `Cracked`: 19,939), inverse-frequency class weights (`tensor([0.7107, 1.6867])`) are applied to the cross-entropy loss function.

### 2. Two-Stage Detection with Segmentation Safeguard
* **Classification (`CrackCNN` & `MobileNetV2`):** Evaluates surface patches to predict `CRACKED` vs. `NON-CRACKED` with softmax confidence scores (and fine-tuned `MobileNetV2`).
* **Segmentation Override Safeguard (`analyze_concrete_image_safeguarded`):** If the classifier misses a subtle or textured crack (such as on brick masonry), the pixel-level segmentation model overrides a false negative (`CRACKED (OVERRIDDEN BY SEGMENTATION)`) whenever a valid crack mask is detected.

### 3. Geometric Severity Quantification
Predicted binary masks are post-processed to extract morphological properties and assign a structural severity level (`LOW`, `MEDIUM`, `HIGH`):
* **Crack Area (px):** Total foreground segmented pixel count.
* **Crack Length (px):** Skeletonized medial-axis centerline length.
* **Average & Maximum Width (px):** Distance transform measurements across the segmented crack region.

---

## 📊 Sample Results & Screenshots

End-to-end Concrete Crack Analyzer outputs (including the original image, predicted binary mask, and centerline localization overlay) are saved in `screenshots/`:

| Capture File | Target Surface | Detection Status | Area | Length | Avg / Max Width | Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[`screenshots/output1.jpeg`](screenshots/output1.jpeg)** | `cracked-concrete-wall_23-2148399198 (1).avif` | `CRACKED` (Segmentation Override, 100.00%) | 1,182.0 px | 279.52 px | 4.57 px / 8.94 px | **HIGH** |
| **[`screenshots/output2.jpeg`](screenshots/output2.jpeg)** | `DSC08006.jpg` (Brick Masonry) | `CRACKED` (Segmentation Override, 100.00%) | 82.0 px | 43.90 px | 2.06 px / 2.83 px | **LOW** |
| **[`screenshots/output3.jpeg`](screenshots/output3.jpeg)** | `original.jpg` (Concrete Surface) | `CRACKED` (99.98% confidence) | 2,095.0 px | 508.87 px | 4.42 px / 8.49 px | **HIGH** |

> *Note: Preliminary inspection assistance only. Does not replace professional structural assessment.*

---

## 👥 Contributors

* **Vaidehi**
* **Shreyas**
* **Vaishnavi**
* **Swanandi**

---

## 📄 License

This project is intended for academic and research purposes. Please refer to the individual Kaggle dataset pages for their respective data usage licenses.
