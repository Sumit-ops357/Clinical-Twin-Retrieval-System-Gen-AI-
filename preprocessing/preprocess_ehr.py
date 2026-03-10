import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    scaler = StandardScaler()
    features = scaler.fit_transform(df)
    return features
