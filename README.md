# Pattern Recognition: Automated Face Recognition

An end-to-end pattern-recognition pipeline for identifying subjects from grayscale face images. The project combines image preprocessing, Haar Wavelet Transform feature extraction, Principal Component Analysis, and multiple classical machine-learning classifiers.

> **Project status:** The training and evaluation pipeline, result artifacts, reports, and presentation materials are included. The generated NumPy dataset is stored separately from the GitHub source repository.

## Project Goal

Face recognition is a supervised classification problem in which each image must be assigned to the correct subject identity. The same person can appear with changes in facial expression, illumination, or head pose, so the system must learn useful visual patterns rather than memorize a single image.

This project implements a transparent pipeline that makes each stage visible:

```text
Face images
    |
    v
Dataset preparation and labels
    |
    v
2D Haar Discrete Wavelet Transform
    |
    v
PCA dimensionality reduction
    |
    v
SVM / KNN / Random Forest training
    |
    v
Accuracy, precision, recall, F1-score
    |
    v
Comparison charts and confusion matrix
```

## Dataset

The project is designed around the Olivetti / AT&T Faces dataset. The preparation script downloads the dataset through scikit-learn, extracts grayscale face images and subject labels, and saves them locally as:

```text
data/
├── X.npy
└── y.npy
```

The dataset contains face images represented as arrays and a target label for each subject. The generated `.npy` files are kept outside the GitHub repository because they are data artifacts rather than source code. Add the Google Drive link for the dataset package to this README when the data has been uploaded.

## Methodology

### 1. Data Preparation

`data_prep.py` uses `fetch_olivetti_faces` to acquire the face dataset, stores the image matrix in `data/X.npy`, stores the labels in `data/y.npy`, and saves a sample image for visual verification.

### 2. Wavelet Feature Extraction

`train_models.py` applies a single-level two-dimensional discrete wavelet transform using the Haar (`db1`) wavelet. The approximation sub-band, usually referred to as `LL`, is flattened into a feature vector. This representation preserves low-frequency structural information while reducing the raw image representation.

### 3. Dimensionality Reduction

PCA is applied to the wavelet features with the number of components selected to retain 95% of the variance. This reduces the dimensionality of the feature matrix and can make the classifiers more efficient and less sensitive to redundant features.

### 4. Classification

The project compares three classifiers:

| Classifier | Role in the experiment |
|---|---|
| SVM with RBF kernel | A strong non-linear classifier for compact, high-dimensional feature vectors. |
| KNN with `k=3` | A simple distance-based baseline. |
| Random Forest | An ensemble classifier based on multiple decision trees. |

### 5. Evaluation

The pipeline calculates Accuracy, Precision, Recall, and F1-Score. It also creates a model-comparison bar chart and a confusion matrix for the best-performing model. The results are saved to `results.json` and the generated image files.

## Repository Structure

```text
.
├── data_prep.py
├── train_models.py
├── data/
│   ├── X.npy                  # Download or generate separately
│   └── y.npy                  # Download or generate separately
├── sample_face.png
├── model_comparison.png
├── confusion_matrix.png
├── results.json
├── presentation_content.md
├── project_plan.md
├── presentation report.pdf
├── AI304 Pattern Recognition Project_ Automated Face Recognition.pptx
├── .gitignore
└── README.md
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install numpy PyWavelets scikit-learn matplotlib seaborn
```

## Prepare the Dataset

To download the Olivetti dataset and create the local NumPy files, run:

```bash
python data_prep.py
```

The script creates the `data/` directory and writes `X.npy` and `y.npy`. If you received a prepared data archive through Google Drive, extract it into the project root instead.

Expected result:

```text
data/X.npy
data/y.npy
```

## Train and Evaluate the Models

After the data files are available, run:

```bash
python train_models.py
```

The script will:

1. Load the image matrix and labels.
2. Apply the 2D Haar wavelet transform.
3. Reduce the transformed features with PCA.
4. Create a stratified train/test split.
5. Train SVM, KNN, and Random Forest models.
6. Print the main evaluation metrics.
7. Save the comparison chart, confusion matrix, and JSON results.

The output files are written to the project root unless you change the paths in `train_models.py`.

## Interpreting the Results

The comparison chart shows how the classifiers perform across Accuracy, Precision, Recall, and F1-Score. The confusion matrix shows which subject classes are confused with one another. A high overall accuracy can still hide weaknesses for individual subjects, so the class-level errors should be reviewed before drawing conclusions.

The repository also includes a presentation and report that explain the methodology and summarize the experimental results. Treat the reported numbers as results for the particular split and preprocessing configuration used by the project, not as a universal benchmark.

## Reproducibility Notes

The project uses a fixed random seed in the dataset preparation and train/test split steps. For a rigorous comparison, keep the same preprocessing, split, and model parameters. When changing the dataset or using a different split, regenerate all evaluation artifacts and update `results.json`.

## Limitations

- The dataset is relatively small compared with modern face-recognition benchmarks.
- The method relies on grayscale images and handcrafted features rather than learned deep visual representations.
- The current repository does not include a webcam or real-time recognition interface.
- Results depend on the subject split, image conditions, and preprocessing choices.
- The system is an academic experiment and should not be used for security or identity decisions without additional validation, privacy review, and bias assessment.

## Future Work

Potential extensions include evaluating on a subject-independent protocol, comparing against CNN embeddings, adding image normalization and alignment, tuning the PCA variance threshold, adding cross-validation, exporting trained models for inference, and implementing a real-time camera demonstration.

## Responsible Use

Face-recognition systems can affect privacy and individuals' rights. Use only datasets with appropriate permissions, avoid uploading identifiable personal images without consent, and do not use this academic project for surveillance, access control, or other high-impact decisions without a complete safety, fairness, and legal review.

## License and Dataset Notice

Check the license and terms of the external face dataset before redistribution. The repository code and project documentation are intended for education and experimentation.
