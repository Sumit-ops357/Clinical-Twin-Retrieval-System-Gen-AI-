import urllib.request
import zipfile
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

url = "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip"
zip_path = "d:/Patient_Similarity_Analysis/data/synthea/synthea_sample_data.zip"
extract_path = "d:/Patient_Similarity_Analysis/data/synthea/"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete. Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete.")
os.remove(zip_path)
