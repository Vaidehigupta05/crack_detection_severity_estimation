# Surface Crack Detection and Segmentation

An end-to-end computer vision pipeline built with **PyTorch** and **Google Colab** for detecting structural surface cracks, generating pixel-level segmentation masks, and performing quantitative severity evaluations (crack area, length, and width).


## Project Overview
This repository contains a dual-purpose pipeline:
1. **Binary Classification:** Quickly screens surface patches to determine if a crack is present (Cracked vs. Non-Cracked).
2. **Pixel-Level Segmentation & Analysis:** Isolates the exact crack pixels using mask annotations to calculate geometric properties (Area, Length, Average/Maximum Width) and estimate structural severity (**LOW / MEDIUM / HIGH**).


## Datasets

This project integrates three public Kaggle datasets:
* **Dataset 1:** [geadalfa/cracked-non-cracked-surface-datasets](https://www.kaggle.com/datasets/geadalfa/cracked-non-cracked-surface-datasets) (Used for binary classification training - ~96K images).
* **Dataset 2:** [aravindnagarajan/crack-segmentation-dataset](https://www.kaggle.com/datasets/aravindnagarajan/crack-segmentation-dataset) (Used for pixel-level image/mask segmentation pairs).
* **Dataset 3:** [arunrk7/surface-crack-detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection) (Balanced positive/negative validation sets).

### Automated Download via KaggleHub
The code automatically downloads and sets up these datasets via `kagglehub`. Ensure you have your Kaggle API credentials configured if running locally, or execute inside a Google Colab runtime.



## Installation & Requirements

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Vaidehigupta05/crack_detection_severity_estimation.git](https://github.com/Vaidehigupta05/crack_detection_severity_estimation.git)
   cd crack_detection_severity_estimation
