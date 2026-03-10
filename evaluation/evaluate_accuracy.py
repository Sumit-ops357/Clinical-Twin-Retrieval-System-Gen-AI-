import sys
import os
import io

# Redirect stdout to suppress prints from imported modules during loading
old_stdout = sys.stdout
sys.stdout = io.StringIO()

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.synthea_preprocessing import load_and_preprocess_synthea
from retrieval.similarity_search import load_embeddings_and_ids

# Restore stdout
sys.stdout = old_stdout

def evaluate_retrieval_accuracy(num_queries=100, top_k=5):
    print("Initializing Accuracy Evaluation...")
    print("1. Loading raw patient data and extracting features...")
    
    # Temporarily suppress output of load_and_preprocess_synthea
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    X, patient_ids_from_data = load_and_preprocess_synthea()
    sys.stdout = old_stdout
    
    print("2. Loading generated embeddings...")
    embeddings, subject_ids = load_embeddings_and_ids()
    
    # Map subject ID to their raw feature vector
    id_to_features = {pid: X[idx] for idx, pid in enumerate(patient_ids_from_data)}
    
    print(f"\nEvaluating Top-{top_k} Retrieval Accuracy over {num_queries} random patients...")
    
    np.random.seed(42)
    query_indices = np.random.choice(len(embeddings), min(num_queries, len(embeddings)), replace=False)
    
    total_exact_matches = 0
    total_comparisons = 0
    feature_match_percentages = []
    
    for q_idx in query_indices:
        query_id = subject_ids[q_idx]
        if query_id not in id_to_features:
            continue
            
        query_features = id_to_features[query_id]
        query_embedding = embeddings[q_idx].reshape(1, -1)
        
        # Calculate similarity in embedding space
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top-k indices (skip self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        for t_idx in top_indices:
            twin_id = subject_ids[t_idx]
            if twin_id not in id_to_features:
                continue
                
            twin_features = id_to_features[twin_id]
            
            # Compare raw features between Query and Twin
            # Calculate what percentage of the features are exactly the same
            matches = np.sum(query_features == twin_features)
            match_percentage = matches / len(query_features)
            
            feature_match_percentages.append(match_percentage)
            
            if match_percentage == 1.0:
                total_exact_matches += 1
            total_comparisons += 1
            
    avg_match_percent = np.mean(feature_match_percentages) * 100
    exact_match_ratio = (total_exact_matches / total_comparisons) * 100
    
    print("-" * 50)
    print("RESULTS: Patient Similarity Feature Overlap Accuracy")
    print("-" * 50)
    print(f"Average Feature Match: {avg_match_percent:.2f}%")
    print(f"Exact Clone Percentage: {exact_match_ratio:.2f}%")
    print("\nNote: Because we currently only use Demographic features (Age, Gender, Race),")
    print("the autoencoder finds exact clones easily. We need to add Medical Conditions")
    print("to the preprocessing to get more nuanced, real-world clinical accuracy!")

if __name__ == "__main__":
    evaluate_retrieval_accuracy()
