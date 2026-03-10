import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def load_embeddings(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    return np.load(filepath)

def plot_pca(embeddings, save_path="evaluation/pca_plot.png"):
    print("Running PCA for visualization...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(x=pca_result[:,0], y=pca_result[:,1], alpha=0.5)
    plt.title('Clinical Twin Latent Space (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(save_path)
    print(f"PCA plot saved to {save_path}")

def plot_tsne(embeddings, save_path="evaluation/tsne_plot.png"):
    print("Running t-SNE for visualization (This might take a moment)...")
    # For large datasets, taking a subset is recommended
    subset = embeddings[:min(5000, len(embeddings))]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(subset)

    plt.figure(figsize=(10, 8))
    plt.scatter(x=tsne_result[:,0], y=tsne_result[:,1], alpha=0.6)
    plt.title('Clinical Twin Latent Space (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")

if __name__ == '__main__':
    try:
        # Create output dir
        os.makedirs('evaluation', exist_ok=True)
        
        # Load and plot
        embeddings = load_embeddings('models/saved/patient_embeddings.npy')
        plot_pca(embeddings)
        plot_tsne(embeddings)
        print("Evaluation complete.")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
