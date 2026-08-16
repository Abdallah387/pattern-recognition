import numpy as np
import pywt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
X = np.load('data/X.npy')
y = np.load('data/y.npy')

# 1. Feature Extraction: Discrete Wavelet Transform (DWT)
def apply_dwt(images):
    dwt_features = []
    for img in images:
        # Use 'db1' (Haar) wavelet, 2D DWT
        coeffs2 = pywt.dwt2(img, 'db1')
        LL, (LH, HL, HH) = coeffs2
        # Use the approximation coefficients (LL) as features
        dwt_features.append(LL.flatten())
    return np.array(dwt_features)

print("Applying DWT...")
X_dwt = apply_dwt(X)
print(f"DWT features shape: {X_dwt.shape}")

# 2. Dimensionality Reduction: PCA
# We want to reduce to a reasonable number of components that explain most variance
pca = PCA(n_components=0.95, whiten=True, random_state=42)
X_pca = pca.fit_transform(X_dwt)
print(f"PCA features shape: {X_pca.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model Training and Comparison
models = {
    "SVM (RBF)": SVC(kernel='rbf', C=10, gamma=0.001, random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# 4. Evaluation and Visualization
# Print metrics table
print("\nModel Comparison Results:")
print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 70)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1-Score']:<10.4f}")

# Plotting metrics comparison
plt.figure(figsize=(10, 6))
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics_names))
width = 0.2

for i, (name, metrics) in enumerate(results.items()):
    vals = [metrics[m] for m in metrics_names]
    plt.bar(x + i*width, vals, width, label=name)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison - Face Recognition')
plt.xticks(x + width, metrics_names)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')

# Plot Confusion Matrix for the best model (usually SVM for this dataset)
best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
plt.figure(figsize=(12, 10))
sns.heatmap(results[best_model_name]['Confusion Matrix'], annot=False, cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')

# Save results for report
import json
# Convert numpy arrays to lists for JSON serialization
json_results = {}
for name, metrics in results.items():
    json_results[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}

with open('results.json', 'w') as f:
    json.dump(json_results, f)

print("\nTraining and comparison complete. Visualizations saved.")
