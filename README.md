# Patient Similarity Analysis (Clinical Twin Engine)

A deep learning-based retrieval system designed to mathematically embed patient Electronic Health Records (EHR) into a latent space using an Autoencoder. The system finds "Clinical Twins" – patients with the most similar demographic and clinical profiles – using Cosine Similarity. 

This project currently leverages the Synthea synthetic patient dataset, extracting demographics and medical conditions (via SNOMED codes) to build highly personalized patient representations.

## Features

- **FHIR Preprocessing Pipeline**: Automatically downloads and parses synthetic patient FHIR JSON bundles (via `kagglehub`).
- **Rich Feature Extraction**: Processes and encodes Demographics (Age, Gender, Race, Ethnicity) and Clinical Conditions into a machine-readable feature matrix using pandas and scikit-learn (`MultiLabelBinarizer`).
- **Deep Autoencoder**: A custom TensorFlow/Keras neural network that compresses high-dimensional patient data into a dense 64-dimensional latent embedding.
- **Similarity Search Engine**: Calculates the cosine similarity between a query patient's embedding and the rest of the database to instantly retrieve the top 5 most similar patients.
- **Evaluation Suite**: Includes scripts to mathematically evaluate and verify feature overlap accuracy between a patient and their retrieved twins.

## Project Structure

```bash
Patient_Similarity_Analysis/
├── data/
│   └── processed/          # Saved feature matrices and IDs
├── evaluation/
│   ├── evaluate_accuracy.py # Script used to calculate feature overlap accuracy
│   └── evaluate_model.py    # Plotting utilities for PCA/t-SNE latent space visualizations
├── models/
│   ├── autoencoder.py       # Autoencoder neural network architecture
│   └── saved/               # Saved model weights (.keras) and generated embeddings (.npy)
├── preprocessing/
│   ├── mimic_preprocessing.py   # Legacy/alternative MIMIC-III data processor
│   ├── preprocess_ehr.py        # Central preprocessing execution script
│   └── synthea_preprocessing.py # Synthea dataset FHIR parsing and feature extraction
├── retrieval/
│   └── similarity_search.py     # Nearest-neighbor Cosine Similarity semantic search engine
├── training/
│   ├── train_autoencoder.py     # Training script intended for MIMIC dataset
│   └── train_model.py           # Training pipeline utilizing Synthea data extraction
└── README.md                    # This file
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- TensorFlow
- Scikit-Learn
- Pandas
- Numpy
- Kagglehub (for grabbing the dataset)

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd Patient_Similarity_Analysis
   ```
2. Set up your virtual environment and install the required packages (create a `requirements.txt` if necessary based on the prerequisites above).

## Usage Guide

To run the full end-to-end pipeline, use the following commands from the root directory (`Patient_Similarity_Analysis`):

### 1. Train the Model & Generate Embeddings
This script will automatically download the required Synthea FHIR dataset, parse the demographics and medical conditions, construct the Autoencoder, train it, and save the resulting model and embeddings.
```bash
python -m training.train_model
```

### 2. Search for Clinical Twins
Once the embeddings are saved, interact with the retrieval engine. This script will randomly pick a patient from your dataset and output their top 5 closest clinical twins along with the Cosine Similarity scores.
```bash
python -m retrieval.similarity_search
```

### 3. Evaluate Engine Accuracy
Curious about the actual mathematical overlap between patients? Run the evaluation script. It analyzes 100 randomly queried patients, calculating the exact percentage of medical features they share with their retrieved twins.
```bash
python evaluation/evaluate_accuracy.py
```

## How It Works

1. **Preprocessing**: The code reads through raw JSON FHIR bundles, plucking out age, gender, race, and an array of SNOMED condition codes. `MultiLabelBinarizer` translates these diverse arrays of conditions into a massive binary feature matrix (e.g., Column `COND_Essential Hypertension` = 1 or 0).
2. **Autoencoder**: The deep neural network acts as a bottleneck. It takes hundreds of sparse binary features and is forced to compress them down to 64 continuous numbers (the embedding) while trying to accurately reconstruct the original features. This forces the model to learn the intrinsic, underlying patterns of patient health.
3. **Retrieval**: By comparing the 64-dimensional embeddings using Cosine Similarity, the system bypasses string-matching entirely, instantly fetching patients who occupy the same semantic "health space."

## Important Note regarding Data
This repository includes a `.gitignore` which prevents the upload of large `.npy` matrices, `.keras` weight files, and cache directories created by standard execution. You must run `train_model.py` locally to populate the `models/saved` directory.
