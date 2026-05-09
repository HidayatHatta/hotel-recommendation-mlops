import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# 1. PERUBAHAN: Import root_mean_squared_error
from sklearn.metrics import root_mean_squared_error 
import mlflow
import mlflow.sklearn

def train_model():
    print("Memuat data yang sudah diproses...")
    df = pd.read_csv('data/processed_hotel_data.csv')
    
    # Fitur dan Target
    X = df[['user_id', 'hotel_id']]
    y = df['rating_review']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow tracking
    mlflow.set_experiment("Hotel_Recommendation_RealData")
    with mlflow.start_run():
        n_estimators = 50
        max_depth = 10 # Dibatasi agar komputasi efisien untuk MVP
        
        print(f"Memulai training model dengan {n_estimators} trees...")
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        print("Melakukan evaluasi...")
        predictions = model.predict(X_test)
        
        # 2. PERUBAHAN: Menggunakan fungsi root_mean_squared_error langsung
        rmse = root_mean_squared_error(y_test, predictions)
        
        # Log parameter dan metrik
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        
        # Log model
        mlflow.sklearn.log_model(model, "hotel_rf_model")
        
        print(f"Training Selesai! Model berhasil dilatih dengan RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_model()