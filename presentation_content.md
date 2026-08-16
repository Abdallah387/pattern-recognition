# AI304 Pattern Recognition Project: Automated Face Recognition

## Slide 1: Title Slide
- **Title:** AI304 Pattern Recognition Project
- **Subtitle:** Automated Face Recognition using DWT, PCA, and Machine Learning
- **Presented by:** Manus AI
- **Date:** May 10, 2026

## Slide 2: Introduction & Problem Statement
- **What is Pattern Recognition?** Enabling computers to identify patterns, classify data, and make intelligent decisions.
- **Project Goal:** Design, implement, and present an applied Pattern Recognition system for face recognition.
- **Problem:** Accurately identify individuals from facial images despite variations (expression, lighting, pose).
- **Motivation:** Critical for security, access control, human-computer interaction.

## Slide 3: Dataset Description
- **Dataset:** Olivetti Research Laboratory Face Database (AT&T Database of Faces).
- **Characteristics:** 400 grayscale images, 40 distinct individuals (10 images/person), 64x64 pixels.
- **Variations:** Expression, lighting, head pose.
- **Preprocessing:** Standardized images, no extensive cleaning required.

## Slide 4: Methodology Overview
- **Complete Pipeline:** Data Acquisition -> Preprocessing -> Feature Extraction -> Dimensionality Reduction -> Classification -> Evaluation.
- **Key Techniques:** Discrete Wavelet Transform (DWT) & Principal Component Analysis (PCA).
- **Classifiers:** Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Random Forest.

## Slide 5: Feature Extraction: Discrete Wavelet Transform (DWT)
- **Purpose:** Decompose images into frequency components.
- **Method:** Single-level 2D DWT with 'db1' (Haar) wavelet.
- **Output:** Approximation coefficients (LL sub-band) as flattened 1D vectors.
- **Benefit:** Captures structural information, reduces noise.

## Slide 6: Dimensionality Reduction: Principal Component Analysis (PCA)
- **Purpose:** Reduce high dimensionality while preserving variance.
- **Method:** Applied to DWT features, retaining components explaining 95% variance.
- **Benefit:** Mitigates curse of dimensionality, creates "eigenfaces".

## Slide 7: Classification Models
- **Support Vector Machine (SVM):** RBF kernel, effective in high-dimensional spaces.
- **K-Nearest Neighbors (KNN):** Non-parametric, classifies based on `k` nearest neighbors.
- **Random Forest Classifier:** Ensemble method, multiple decision trees, high accuracy.

## Slide 8: Model Comparison Metrics
- **Metrics:** Accuracy, Precision, Recall, F1-Score.
- **Results Table:**

| Model           | Accuracy | Precision | Recall | F1-Score |
| :-------------- | :------- | :-------- | :----- | :------- |
| SVM (RBF)       | 0.9500   | 0.9667    | 0.9500 | 0.9467   |
| KNN (k=3)       | 0.8125   | 0.8083    | 0.8125 | 0.7842   |
| Random Forest   | 0.9500   | 0.9708    | 0.9500 | 0.9483   |

## Slide 9: Visualizations: Model Comparison
- **Chart:** Bar chart comparing Accuracy, Precision, Recall, F1-Score for all models.
- **Insight:** SVM and Random Forest significantly outperform KNN.
- **Image:** `model_comparison.png`

## Slide 10: Visualizations: Confusion Matrix (Random Forest)
- **Purpose:** Detailed breakdown of classification performance per class.
- **Insight:** Random Forest shows excellent performance with minimal misclassifications.
- **Image:** `confusion_matrix.png`

## Slide 11: Conclusion & Future Improvements
- **Summary:** Successful implementation of face recognition using DWT, PCA, SVM, KNN, and Random Forest.
- **Key Finding:** SVM and Random Forest achieved 95% accuracy, with Random Forest showing a slight edge.
- **Limitations & Future Work:** Explore advanced feature extraction (CNNs), extensive hyperparameter tuning, evaluation on diverse datasets (LFW), real-time optimization, and improved robustness to variations.

## Slide 12: Q&A
- **Thank You!**
- **Questions?**
