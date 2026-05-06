import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.synthea_preprocessing import load_and_preprocess_synthea

def evaluate_reconstruction_loss():
    print("Initializing Reconstruction Loss Evaluation...")
    
    print("1. Loading Autoencoder and Scaler...")
    try:
        autoencoder = tf.keras.models.load_model('models/saved/clinical_twin_autoencoder.keras')
        scaler = joblib.load('models/saved/scaler.pkl')
    except Exception as e:
        print(f"Error loading models. Have you run train_model.py recently? Error: {e}")
        return

    print("2. Loading raw patient data...")
    # Load raw data
    X, _ = load_and_preprocess_synthea()
    
    # Normalize using the loaded training scaler
    X_scaled = scaler.transform(X)
    
    # We must recreate the exact same train/test split from train_model.py
    # Random state 42 ensures we test on the exact same unseen 20%
    _, X_test = train_test_split(
        X_scaled,
        test_size=0.2,
        random_state=42
    )

    print(f"3. Reconstructing data for {X_test.shape[0]} test patients through the Autoencoder...")
    # This represents: Original Data -> Encoder -> Decoder -> Reconstructed Data
    X_test_reconstructed = autoencoder.predict(X_test, verbose=0)
    
    print("\n4. Calculating Mean Squared Error (MSE)...")
    # This measures the exact average difference between Output and Original Input
    mse = mean_squared_error(X_test, X_test_reconstructed)
    
    print("-" * 60)
    print("RESULTS: Autoencoder Reconstruction Loss")
    print("-" * 60)
    print(f"Mean Squared Error (MSE) on Unseen Test Data: {mse:.4f}")
    
    # Optionally, we can also look at average absolute difference
    mae = np.mean(np.abs(X_test - X_test_reconstructed))
    print(f"Mean Absolute Error (MAE) on Unseen Test Data: {mae:.4f}")
    print("-" * 60)
    
    with open('final_mse.txt', 'w') as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\n")

if __name__ == "__main__":
    evaluate_reconstruction_loss()
