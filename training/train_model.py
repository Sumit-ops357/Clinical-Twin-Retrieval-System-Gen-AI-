from preprocessing.synthea_preprocessing import load_and_preprocess_synthea
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.autoencoder import build_autoencoder

# Load dataset
X, patient_ids = load_and_preprocess_synthea()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test = train_test_split(
    X_scaled,
    test_size=0.2,
    random_state=42
)

# Build model
autoencoder, encoder = build_autoencoder(X_train.shape[1])

# Train
autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, X_test)
)

import os
import numpy as np

# Generate embeddings on all data
print("Generating embeddings (latent representations)...")
embeddings = encoder.predict(X_scaled) 

# Save the encoder for inference and embeddings for retrieval
os.makedirs('models/saved', exist_ok=True)
encoder.save('models/saved/clinical_twin_encoder.keras') # Use .keras extension as recommended in newer TF versions
autoencoder.save('models/saved/clinical_twin_autoencoder.keras') # Saving full model for Reconstruction evaluation
np.save('models/saved/patient_embeddings.npy', embeddings)
np.save('models/saved/synthea_subject_ids.npy', patient_ids)

import joblib
joblib.dump(scaler, 'models/saved/scaler.pkl')

print("Training Completed. Model and Embeddings saved successfully.")