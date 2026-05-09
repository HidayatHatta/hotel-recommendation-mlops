import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def process_data():
    file_path = 'data/raw_hotel_reviews.csv'
    
    print("Membaca dataset...")
    # on_bad_lines='skip' : jika ada baris CSV yang formatnya rusak
    df = pd.read_csv(file_path, sep=';', on_bad_lines='skip', low_memory=False)
    
    # Memilih kolom yang relevan untuk sistem rekomendasi
    df = df[['user_name', 'hotel_name', 'rating_review']].dropna()
    
    print(f"Total data bersih: {len(df)} baris. Memulai encoding...")
    
    # Inisialisasi Encoder
    user_encoder = LabelEncoder()
    hotel_encoder = LabelEncoder()
    
    # Transformasi string menjadi integer ID
    df['user_id'] = user_encoder.fit_transform(df['user_name'].astype(str))
    df['hotel_id'] = hotel_encoder.fit_transform(df['hotel_name'].astype(str))
    
    # Pastikan folder model untuk API tersedia
    os.makedirs('api/model', exist_ok=True)
    
    # Simpan encoder sebagai artifact model (Penting untuk Production!)
    joblib.dump(user_encoder, 'api/model/user_encoder.pkl')
    joblib.dump(hotel_encoder, 'api/model/hotel_encoder.pkl')
    
    # Simpan data yang siap di-training
    final_df = df[['user_id', 'hotel_id', 'rating_review']]
    final_df.to_csv('data/processed_hotel_data.csv', index=False)
    
    print("Data processing selesai. Data disimpan di data/processed_hotel_data.csv")

if __name__ == "__main__":
    process_data()