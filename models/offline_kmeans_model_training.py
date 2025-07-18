#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans

# ========== CONFIGURATION ==========
input_dataset_path = "/Users/omar/Desktop/application/First year/Scripts/BCCC-CIC-Bell-DNS-2024/Test/offline_training_data_filtered.csv"
model_save_path = "kmeans_model.joblib"

k = 279
random_state = 42

def main():
    # Validate that the input dataset file exists.
    if not os.path.exists(input_dataset_path):
        print(f"Error: Input dataset file '{input_dataset_path}' not found.")
        return
    
    try:
        print("Loading offline dataset...")
        data = pd.read_csv(input_dataset_path)
        print(f"Dataset loaded with shape: {data.shape}")
    except Exception as e:
        print(f"Error reading dataset file '{input_dataset_path}': {e}")
        return
    
    # Drop the ground truth label (if it exists)
    if 'label' in data.columns:
        data = data.drop(columns=['label'])
        print("Dropped 'label' column for KMeans training.")
    
    # Validate that the dataset is not empty after dropping columns
    if data.empty:
        print("Error: The dataset is empty after processing.")
        return

    # Convert the DataFrame to a NumPy array (forcing a float dtype for KMeans)
    try:
        X = data.values.astype(float)
    except Exception as e:
        print(f"Error converting dataset to a numeric array: {e}")
        return
    
    # Warn if the model file already exists; it will be overwritten.
    if os.path.exists(model_save_path):
        print(f"Warning: '{model_save_path}' already exists and will be overwritten.")
    
    # Fit the KMeans model
    try:
        print(f"Fitting KMeans model with k={k} and random_state={random_state}...")
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
    except Exception as e:
        print(f"Error during KMeans fitting: {e}")
        return
    
    print(f"Model trained with {kmeans.n_clusters} clusters. Centroid shape: {kmeans.cluster_centers_.shape}")
    
    # Save the trained KMeans model using joblib.dump
    try:
        print(f"Saving trained KMeans model to '{model_save_path}'...")
        joblib.dump(kmeans, model_save_path)
        print("KMeans model saved successfully.")
    except Exception as e:
        print(f"Error saving KMeans model: {e}")

if __name__ == "__main__":
    main()
