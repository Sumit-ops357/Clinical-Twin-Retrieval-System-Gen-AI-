import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_and_evaluate_classifier(X, y):
    print("Training classifier with provided data...")

    # Split the dataset
    print("\nSplitting into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning
    print("Performing hyperparameter tuning using GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    clf = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    # Train the best model
    print(f"Training the best Random Forest Classifier on {len(X_train)} patients...")
    clf.fit(X_train, y_train)

    # Predict and evaluate
    print(f"Evaluating on Test Set ({len(X_test)} patients)...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    correct_preds = sum(y_pred == y_test)
    total_preds = len(y_test)

    print("\n" + "="*50)
    print("SUPERVISED CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Total Test Patients: {total_preds}")
    print(f"Correct Predictions: {correct_preds}")
    print(f"Accuracy = Correct / Total = {acc*100:.2f}%")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("="*50)

    return clf

if __name__ == '__main__':
    # Load and preprocess data
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
    
    # Train and evaluate
    clf = train_and_evaluate_classifier(X, y)
    
    print(f"Training complete. Model saved to 'evaluation/model.pkl'")
    # Save the model
    import pickle
    with open('evaluation/model.pkl', 'wb') as f:
        pickle.dump(clf, f)
