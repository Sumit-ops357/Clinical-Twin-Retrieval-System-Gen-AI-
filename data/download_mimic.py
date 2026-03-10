import urllib.request
import zipfile
import os

url = "https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip"
zip_path = "d:/Patient_Similarity_Analysis/data/mimic/mimic-demo.zip"
extract_path = "d:/Patient_Similarity_Analysis/data/mimic/"

print(f"Downloading {url}...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete. Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete.")
# Optional cleanup
os.remove(zip_path)
