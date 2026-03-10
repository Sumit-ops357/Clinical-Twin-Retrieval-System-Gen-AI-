import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.synthea_preprocessing import load_and_preprocess_synthea

def train_and_evaluate_classifier():
    print("Loading Synthea Data as DataFrame...")
    df = load_and_preprocess_synthea(max_patients=2000, return_df=True)
    
    # Let's find a target condition to predict.
    condition_cols = [c for c in df.columns if c.startswith('COND_')]
    
    # Find condition with highest variance (closest to 50% split) or just pick a common one
    best_cond = None
    best_variance = -1
    for col in condition_cols:
        mean_val = df[col].astype(float).mean()
        if 0.1 < mean_val < 0.9: # Between 10% and 90% prevalence
            variance = mean_val * (1 - mean_val)
            if variance > best_variance:
                best_variance = variance
                best_cond = col
                
    if not best_cond:
        print("No condition with 10-90% prevalence found. Picking the most prevalent one.")
        best_cond = df[condition_cols].astype(float).mean().idxmax()
        
    condition_name = best_cond.replace('COND_', '')
    print(f"\nTarget Variable selected: '{condition_name}'")
    print(f"Prevalence in dataset: {df[best_cond].mean()*100:.2f}%")
    
    # Prepare X and y
    y = df[best_cond].astype(int).values
    X_df = df.drop(columns=[best_cond])
    X = X_df.astype(float).values
    
    print(f"Remaining Features used for prediction: {X.shape[1]}")
    
    # Split the dataset
    print("\nSplitting into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print(f"Training Random Forest Classifier on {len(X_train)} patients...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    print(f"Evaluating on Test Set ({len(X_test)} patients)...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    correct_preds = sum(y_pred == y_test)
    total_preds = len(y_test)
    
    print("\n" + "="*50)
    print("SUPERVISED CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Target Condition: {condition_name}")
    print(f"Total Test Patients: {total_preds}")
    print(f"Correct Predictions: {correct_preds}")
    print(f"Accuracy = Correct / Total = {acc*100:.2f}%")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Export predictions to CSV for manual verification
    print("\nExporting predictions to CSV for manual check...")
    results_df = pd.DataFrame({
        'Actual_Condition': y_test,
        'Predicted_Condition': y_pred,
        'Is_Correct': (y_test == y_pred)
    })
    
    results_df.to_csv('evaluation/manual_accuracy_check.csv', index=False)
    print(f"Saved {len(results_df)} test predictions to 'evaluation/manual_accuracy_check.csv'")
    print("You can open this file in Excel to manually count the 'True' values in Is_Correct!")

if __name__ == '__main__':
    train_and_evaluate_classifier()
