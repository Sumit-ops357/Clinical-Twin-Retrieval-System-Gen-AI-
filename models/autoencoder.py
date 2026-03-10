import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_dim=5000, latent_dim=64):
    """
    Builds the Autoencoder model for representation learning.
    5000 dimensional EHR -> Encoder -> 64 latent vector -> Decoder
    """
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(1024, activation='relu')(encoder_inputs)
    x = layers.Dense(256, activation='relu')(x)
    latent_vector = layers.Dense(latent_dim, activation='relu', name='latent_vector')(x)
    
    # Decoder
    x = layers.Dense(256, activation='relu')(latent_vector)
    x = layers.Dense(1024, activation='relu')(x)
    decoder_outputs = layers.Dense(input_dim, activation='linear')(x)
    
    # Models
    autoencoder = models.Model(encoder_inputs, decoder_outputs, name='autoencoder')
    encoder = models.Model(encoder_inputs, latent_vector, name='encoder')
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

if __name__ == '__main__':
    autoencoder, encoder = build_autoencoder()
    autoencoder.summary()