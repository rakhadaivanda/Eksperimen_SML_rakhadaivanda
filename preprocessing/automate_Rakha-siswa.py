import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

def run_preprocessing(input_path, output_folder):
    """
    Fungsi untuk melakukan preprocessing data diabetes secara otomatis.
    """
    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan.")
        return
    
    df = pd.read_csv(input_path)
    print(f"Memproses {len(df)} baris data...")

    # 2. Pembersihan Data (Handling Duplicates)
    df = df.drop_duplicates()

    # 3. Encoding Data Kategorikal
    # Menggunakan LabelEncoder untuk gender dan smoking_history
    le_gender = LabelEncoder()
    le_smoke = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])

    # 4. Feature Scaling
    # Normalisasi fitur numerik agar rentang nilainya seragam
    scaler = StandardScaler()
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 5. Menyiapkan Output Folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 6. Menyimpan Data Hasil Preprocessing
    # Data disimpan dalam format CSV untuk digunakan di tahap modelling (Kriteria 2)
    output_path = os.path.join(output_folder, 'diabetes_cleaned.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessing selesai! Data disimpan di: {output_path}")
    return df

if __name__ == "__main__":
    # Sesuaikan path dengan struktur folder Kriteria 1
    RAW_DATA_PATH = "rakhadataset_raw/diabetes_prediction_dataset.csv"
    PROCESSED_FOLDER = "preprocessing/rakhadataset_preprocessing"
    
    run_preprocessing(RAW_DATA_PATH, PROCESSED_FOLDER)