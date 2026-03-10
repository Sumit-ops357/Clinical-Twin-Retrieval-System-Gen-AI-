import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Define Paths
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'saved'))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))

def load_embeddings_and_ids():
    embeddings_path = os.path.join(MODELS_DIR, 'patient_embeddings.npy')
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("Embeddings not found. Run training first.")
        
    embeddings = np.load(embeddings_path)

    # Try to load corresponding IDs based on embeddings length
    mimic_ids_path = os.path.join(PROCESSED_DATA_DIR, 'mimic_subject_ids.npy')
    synthea_ids_path = os.path.join(MODELS_DIR, 'synthea_subject_ids.npy')
    
    if os.path.exists(synthea_ids_path):
        subject_ids = np.load(synthea_ids_path, allow_pickle=True)
        if len(subject_ids) == len(embeddings):
            return embeddings, subject_ids
            
    if os.path.exists(mimic_ids_path):
        subject_ids = np.load(mimic_ids_path, allow_pickle=True)
        if len(subject_ids) == len(embeddings):
            return embeddings, subject_ids
            
    # Fallback if no matching IDs found
    print(f"Warning: Could not find matching subject IDs (length {len(embeddings)}). Using dummy IDs.")
    subject_ids = np.arange(len(embeddings))
    
    return embeddings, subject_ids

def find_clinical_twins(query_index, embeddings, subject_ids, top_k=5):
    """
    Finds the top-K similar patients to the query patient using cosine similarity.
    """
    if query_index >= len(embeddings) or query_index < 0:
        raise ValueError("Invalid query index.")
        
    query_embedding = embeddings[query_index].reshape(1, -1)
    
    # Compute cosine similarity between query and all other patients
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get indices of top_k + 1 (including the query patient itself, which should be 1.0)
    top_indices = np.argsort(similarities)[::-1][:top_k + 1]
    
    results = []
    print(f"\nQuery Patient SUBJECT_ID: {subject_ids[query_index]}")
    print("-" * 40)
    for idx in top_indices:
        # Skip the query patient itself in the output list since its similarity is always 1
        if idx == query_index:
            continue
        results.append({
            'subject_id': subject_ids[idx],
            'similarity_score': similarities[idx]
        })
        print(f"Twin SUBJECT_ID: {subject_ids[idx]} | Cosine Similarity: {similarities[idx]:.4f}")
        
    # Standardize result size if the array size was exactly top_k
    return results[:top_k]

if __name__ == '__main__':
    try:
        print("Loading representation space...")
        embeddings, subject_ids = load_embeddings_and_ids()
        
        # Test with a random patient index
        sample_query_idx = np.random.randint(0, len(embeddings))
        find_clinical_twins(sample_query_idx, embeddings, subject_ids, top_k=5)
    except FileNotFoundError as e:
        print(e)
