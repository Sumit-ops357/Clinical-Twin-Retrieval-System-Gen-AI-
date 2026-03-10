import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Define paths
DATA_DIR = r"d:\Patient_Similarity_Analysis\data\mimic\mimic-iii-clinical-database-demo-1.4"
PROCESSED_DATA_DIR = r"d:\Patient_Similarity_Analysis\data\processed"

def load_data():
    patients_df = pd.read_csv(os.path.join(DATA_DIR, "PATIENTS.csv"))
    patients_df.columns = patients_df.columns.str.upper()
    admissions_df = pd.read_csv(os.path.join(DATA_DIR, "ADMISSIONS.csv"))
    admissions_df.columns = admissions_df.columns.str.upper()
    labevents_df = pd.read_csv(os.path.join(DATA_DIR, "LABEVENTS.csv"))
    labevents_df.columns = labevents_df.columns.str.upper()
    return patients_df, admissions_df, labevents_df

def extract_features(patients_df, admissions_df, labevents_df):
    print("Extracting features from MIMIC-III Demo...")
    # Get first admission for each patient to compute age
    first_admissions = admissions_df.sort_values(['SUBJECT_ID', 'ADMITTIME']).groupby('SUBJECT_ID').first().reset_index()
    
    # Merge with patients to get DOB
    patient_info = pd.merge(first_admissions[['SUBJECT_ID', 'ADMITTIME']], patients_df[['SUBJECT_ID', 'DOB', 'GENDER']], on='SUBJECT_ID')
    
    # Convert dates to datetime (just get the Year instead of full datetime to avoid Overflow errors)
    patient_info['DOB_YEAR'] = patient_info['DOB'].str[:4].astype(float)
    patient_info['ADMIT_YEAR'] = patient_info['ADMITTIME'].str[:4].astype(float)
    
    # Calculate age at first admission (approximate due to MIMIC shifting)
    patient_info['AGE'] = patient_info['ADMIT_YEAR'] - patient_info['DOB_YEAR']
    # Fix age for >89 years old (MIMIC sets DOB to 1890s for >89yo)
    patient_info.loc[patient_info['AGE'] > 89, 'AGE'] = 90
    
    # Encode gender
    patient_info['GENDER_M'] = (patient_info['GENDER'] == 'M').astype(int)
    patient_info['GENDER_F'] = (patient_info['GENDER'] == 'F').astype(int)
    
    # Keep only relevant demographic columns and SUBJECT_ID
    demographics = patient_info[['SUBJECT_ID', 'AGE', 'GENDER_M', 'GENDER_F']]
    
    # Process lab events (numerical values only)
    labs_num = labevents_df.dropna(subset=['VALUENUM'])
    
    # Pivot lab events: get average lab value per ITEMID per SUBJECT_ID
    lab_features = labs_num.pivot_table(index='SUBJECT_ID', columns='ITEMID', values='VALUENUM', aggfunc='mean').reset_index()
    
    # Merge demographics and lab features
    features_df = pd.merge(demographics, lab_features, on='SUBJECT_ID', how='left')
    
    return features_df

def impute_and_normalize(features_df):
    print("Imputing and normalizing features...")
    # Save SUBJECT_ID as index
    subject_ids = features_df['SUBJECT_ID'].values
    features_df = features_df.drop(columns=['SUBJECT_ID'])
    
    # Ensure all data is numeric to avoid StandardScaler TypeError
    features_df = features_df.astype(float)
    features_df.columns = features_df.columns.astype(str)
    
    # Fill missing values with 0 (or column mean)
    features_df = features_df.fillna(0)
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_df)
    
    return normalized_features, subject_ids

def run_preprocessing():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    patients_df, admissions_df, labevents_df = load_data()
    features_df = extract_features(patients_df, admissions_df, labevents_df)
    normalized_features, subject_ids = impute_and_normalize(features_df)
    
    print(f"Generated feature matrix with shape: {normalized_features.shape}")
    
    # Save data
    np.save(os.path.join(PROCESSED_DATA_DIR, 'mimic_features.npy'), normalized_features)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'mimic_subject_ids.npy'), subject_ids)
    print("Saved feature matrix and subject IDs to folder:")
    print(PROCESSED_DATA_DIR)

if __name__ == '__main__':
    run_preprocessing()
