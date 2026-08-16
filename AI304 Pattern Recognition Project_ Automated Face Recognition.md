# AI304 Pattern Recognition Project: Automated Face Recognition

## Project Overview
This project implements a complete pattern recognition pipeline for automated face recognition. It utilizes the AT&T Database of Faces (Olivetti faces) and applies Discrete Wavelet Transform (DWT) for feature extraction and Principal Component Analysis (PCA) for dimensionality reduction. The project compares three different machine learning models: Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest.

## Project Structure
- `data_prep.py`: Script to download and prepare the Olivetti faces dataset.
- `train_models.py`: Main script to perform feature extraction, dimensionality reduction, model training, and comparison.
- `AI304_Project_Report.md`: Detailed technical report covering all project stages and results.
- `presentation_content.md`: Content used for the presentation slides.
- `sample_face.png`: A sample image from the dataset.
- `model_comparison.png`: Visualization of the performance metrics comparison.
- `confusion_matrix.png`: Confusion matrix for the best-performing model.
- `results.json`: Raw numerical results from the model comparison.

## Requirements
- Python 3.11+
- scikit-learn
- PyWavelets
- numpy
- matplotlib
- seaborn

## Setup and Execution
1. Install the required dependencies:
   ```bash
   pip install scikit-learn PyWavelets numpy matplotlib seaborn
   ```
2. Run the data preparation script:
   ```bash
   python data_prep.py
   ```
3. Run the model training and comparison script:
   ```bash
   python train_models.py
   ```

## Results Summary
- **SVM (RBF):** 95.00% Accuracy
- **Random Forest:** 95.00% Accuracy
- **KNN (k=3):** 81.25% Accuracy

Both SVM and Random Forest proved highly effective for this task, with Random Forest showing a slight edge in precision and F1-score.
