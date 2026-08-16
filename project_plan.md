# Project Proposal: Automated Face Recognition using PCA and Discrete Wavelet Transform (DWT)

## 1. Project Overview
This project aims to build a robust Face Recognition system, a core application of Pattern Recognition. We will implement a complete pipeline from data acquisition to performance evaluation, comparing multiple machine learning models.

## 2. Dataset
- **Dataset:** AT&T Database of Faces (formerly the Olivetti Research Laboratory database) or the `sklearn` Olivetti faces dataset.
- **Details:** 400 images of 40 different subjects (10 images per subject).

## 3. Methodology (Pipeline)
1. **Data Acquisition:** Loading the Olivetti faces dataset.
2. **Preprocessing:** Normalization and resizing.
3. **Feature Extraction & Dimensionality Reduction:**
   - **Wavelet Transform:** Use 2D Discrete Wavelet Transform (DWT) to extract multi-resolution features (approximation coefficients).
   - **PCA (Principal Component Analysis):** Apply PCA on the wavelet-transformed features to reduce dimensionality and retain the most significant "eigenfaces".
4. **Classification:**
   - **Model 1:** Support Vector Machine (SVM) with RBF kernel.
   - **Model 2:** K-Nearest Neighbors (KNN).
   - **Model 3:** Random Forest Classifier.
5. **Evaluation:**
   - Metrics: Accuracy, Precision, Recall, F1-Score.
   - Visualization: Confusion Matrix, Eigenfaces visualization.

## 4. Technical Requirements Fulfillment
- **Complete Pipeline:** Yes.
- **PCA/Wavelet:** Both will be implemented and integrated.
- **Model Comparison:** Yes (SVM vs KNN vs Random Forest).
- **Deliverables:** Source code, Report, Presentation, README.
