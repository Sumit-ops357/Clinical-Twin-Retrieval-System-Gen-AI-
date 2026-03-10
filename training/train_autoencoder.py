import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder import build_autoencoder

def load_real_data():
    features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'mimic_features.npy'))
    if not os.path.exists(features_path):
        raise FileNotFoundError("MIMIC features not found. Please run preprocessing first.")
    return np.load(features_path)

def train_and_save_model():
    print("Loading preprocessed dataset...")
    features = load_real_data()
    
    input_dim = features.shape[1]
    print(f"Dataset Shape: {features.shape}")
    print(f"Training Autoencoder on {features.shape[0]} patients with {input_dim} features...")
    autoencoder, encoder = build_autoencoder(input_dim=input_dim, latent_dim=64)
    
    # Train the model
    autoencoder.fit(
        features, features,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Generate embeddings
    print("Generating embeddings (latent representations)...")
    embeddings = encoder.predict(features)
    
    # Save the encoder for inference and embeddings for retrieval
    os.makedirs('models/saved', exist_ok=True)
    encoder.save('models/saved/clinical_twin_encoder.h5')
    np.save('models/saved/patient_embeddings.npy', embeddings)
    print("Model and Embeddings saved successfully.")

if __name__ == '__main__':
    train_and_save_model()
