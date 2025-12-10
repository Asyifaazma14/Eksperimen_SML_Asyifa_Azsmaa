import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.target_col = "Sleep Disorder"
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()

    def load_data(self, filepath):
        # Cek apakah file ada sebelum membaca
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File tidak ditemukan di path: {filepath}")
            
        self.data = pd.read_csv(filepath)
        print(f"[INFO] Data dimuat dari: {filepath}")
        print(f"[INFO] Shape awal: {self.data.shape}")
        return self.data

    def handle_missing_values(self):
        initial_rows = self.data.shape[0]
        self.data.dropna(inplace=True)
        dropped = initial_rows - self.data.shape[0]
        print(f"[STEP 1] Baris missing dihapus: {dropped}")
        return self.data

    def encode_categorical(self):
        # 1) auto-detect kategori (dtype object)
        categorical_cols = self.data.select_dtypes(include=["object"]).columns.tolist()

        # 2) exclude target dari fitur kategorikal
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)

        # 3) label encoding fitur kategorikal
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
            # print(f"[STEP 2] Label Encoding fitur: {col}")

        # 4) label encoding target khusus (jika target ada di data)
        if self.target_col in self.data.columns:
            self.data[self.target_col] = self.target_encoder.fit_transform(
                self.data[self.target_col]
            )
            print(f"[STEP 2] Label Encoding target: {self.target_col}")

        return self.data

    def scale_numeric(self):
        # 5) auto-detect numerik (int/float)
        numeric_cols = self.data.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # 6) pastikan target gak ikut scaling
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        # 7) scaling numerik
        if numeric_cols:
            self.data[numeric_cols] = self.scaler.fit_transform(self.data[numeric_cols])
            print(f"[STEP 3] Scaling numerik: {numeric_cols}")

        return self.data

    def save_artifacts(self, output_dir):
        # Simpan artifact (model pendukung) ke folder processed juga biar rapi
        # scaler.pkl, label_encoders.pkl, dll
        
        with open(os.path.join(output_dir, "label_encoders.pkl"), "wb") as f:
            pickle.dump(self.label_encoders, f)
        
        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(output_dir, "target_encoder.pkl"), "wb") as f:
            pickle.dump(self.target_encoder, f)
            
        print(f"[INFO] Artifacts (.pkl) disimpan di folder: {output_dir}")

    def save_data(self, output_filepath):
        self.data.to_csv(output_filepath, index=False)
        print(f"[INFO] Data bersih disimpan ke: {output_filepath}")

    def run(self, input_filepath, output_filepath):
        print("--- MEMULAI PIPELINE AUTOMASI (REPLIKA EKSPERIMEN) ---")
        
        # Pastikan folder output ada dulu
        output_dir = os.path.dirname(output_filepath)
        os.makedirs(output_dir, exist_ok=True)

        # Eksekusi tahapan
        self.load_data(input_filepath)
        self.handle_missing_values()
        self.encode_categorical()
        self.scale_numeric()
        
        # Simpan hasil dan artifacts
        self.save_artifacts(output_dir)
        self.save_data(output_filepath)
        print("--- SELESAI ---")

if __name__ == "__main__":
    # --- KONFIGURASI PATH (PENTING!) ---
    # Menggunakan 'os.getcwd()' agar aman dijalankan dari root repository
    base_dir = os.getcwd()
    
    # Path Input (Sesuai struktur folder: data/raw/...)
    # Pastikan nama file CSV di folder raw SAMA PERSIS dengan ini (besar kecil hurufnya)
    INPUT_FILE = os.path.join(base_dir, 'data', 'raw', 'Sleep_health_and_lifestyle_dataset.csv')
    
    # Path Output (Sesuai struktur folder: data/processed/...)
    OUTPUT_FILE = os.path.join(base_dir, 'data', 'processed', 'data_bersih_automasi.csv')

    # Jalankan proses
    try:
        preprocessor = DataPreprocessor()
        preprocessor.run(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
        exit(1) # Keluar dengan kode error agar GitHub Actions tahu kalau ini gagal
