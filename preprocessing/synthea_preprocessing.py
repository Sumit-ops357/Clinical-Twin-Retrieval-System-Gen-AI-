import os
import json
import kagglehub
import pandas as pd

def load_and_preprocess_synthea(max_patients=2000):
    # Set Cache Directory to D drive
    os.environ["KAGGLEHUB_CACHE"] = "d:/kagglehub_cache"
    
    # Download dataset
    path = kagglehub.dataset_download(
        "krsna540/synthea-dataset-jsons-ehr"
    )

    print("Dataset Path:", path)

    # Fast finding of up to max_patients files
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".json"):
                files.append(os.path.join(root, filename))
                if len(files) >= max_patients:
                    break
        if len(files) >= max_patients:
            break
            
    print(f"Parsing {len(files)} patient FHIR bundles...")
    
    patients_data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            patient_info = {}
            patient_conditions = []
            
            for entry in data.get('entry', []):
                resource = entry.get('resource', {})
                res_type = resource.get('resourceType')
                
                if res_type == 'Patient':
                    patient_info['Id'] = resource.get('id')
                    patient_info['GENDER'] = resource.get('gender')
                    patient_info['BIRTHDATE'] = resource.get('birthDate')
                    
                    for ext in resource.get('extension', []):
                        url = ext.get('url', '')
                        if 'us-core-race' in url:
                            try:
                                patient_info['RACE'] = ext['valueCodeableConcept']['coding'][0]['display']
                            except:
                                patient_info['RACE'] = 'Unknown'
                        elif 'us-core-ethnicity' in url:
                            try:
                                patient_info['ETHNICITY'] = ext['valueCodeableConcept']['coding'][0]['display']
                            except:
                                patient_info['ETHNICITY'] = 'Unknown'
                                
                elif res_type == 'Condition':
                    try:
                        # Extract the display name of the condition
                        condition_name = resource['code']['coding'][0]['display']
                        patient_conditions.append(condition_name)
                    except:
                        pass
            
            if patient_info:
                # Add a unique list of conditions to the patient dictionary
                patient_info['CONDITIONS'] = list(set(patient_conditions))
                patients_data.append(patient_info)
        except Exception:
            pass

    patients = pd.DataFrame(patients_data)

    # Convert birthdate -> age
    patients['BIRTHDATE'] = pd.to_datetime(patients['BIRTHDATE'])
    patients['AGE'] = 2025 - patients['BIRTHDATE'].dt.year

    # Encode categorical variables (Demographics)
    patients = pd.get_dummies(
        patients,
        columns=['GENDER','RACE','ETHNICITY']
    )
    
    # Process Medical Conditions using MultiLabelBinarizer
    # Many patients have multiple conditions, this makes a binary column for each unique condition
    from sklearn.preprocessing import MultiLabelBinarizer
    if 'CONDITIONS' in patients.columns:
        mlb = MultiLabelBinarizer()
        condition_matrix = mlb.fit_transform(patients['CONDITIONS'])
        condition_df = pd.DataFrame(condition_matrix, columns=[f"COND_{c}" for c in mlb.classes_])
        
        # Merge conditions back into main dataframe
        patients = pd.concat([patients, condition_df], axis=1)
        
        # Drop the original list column
        patients = patients.drop(columns=['CONDITIONS'])

    # Extract patient IDs before dropping
    if 'Id' in patients.columns:
        patient_ids = patients['Id'].values
    else:
        patient_ids = np.arange(len(patients))

    # Drop unnecessary columns
    patients = patients.drop(
        columns=['Id','BIRTHDATE','DEATHDATE'],
        errors='ignore'
    )

    # Convert to feature matrix
    X = patients.fillna(0).astype(float).values

    print("Train Shape:", X.shape)

    return X, patient_ids