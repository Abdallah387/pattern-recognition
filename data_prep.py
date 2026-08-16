import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import os

def prepare_data():
    print("Fetching Olivetti faces dataset...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.images
    y = data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Save as numpy files for easy access
    os.makedirs('data', exist_ok=True)
    np.save('data/X.npy', X)
    np.save('data/y.npy', y)
    
    # Save a sample image to verify
    plt.imshow(X[0], cmap='gray')
    plt.title(f"Subject: {y[0]}")
    plt.axis('off')
    plt.savefig('sample_face.png')
    print("Data preparation complete. Sample image saved as sample_face.png.")

if __name__ == "__main__":
    prepare_data()
