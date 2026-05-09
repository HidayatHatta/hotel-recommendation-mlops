from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Hotel Recommendation API")

MODEL_URI = "model"
try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception as e:
    model = None
    print(f"Model belum dimuat: {e}")

class RecommendationRequest(BaseModel):
    user_id: int
    hotel_id: int

@app.post("/predict")
def predict_rating(request: RecommendationRequest):
    if model is None:
        return {"error": "Model tidak tersedia."}
    
    data = pd.DataFrame([request.dict()])
    prediction = model.predict(data)
    
    return {
        "user_id": request.user_id,
        "hotel_id": request.hotel_id,
        "predicted_rating": round(float(prediction[0]), 2)
    }
