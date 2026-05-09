# End-to-End Hotel Recommendation MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

## Project Overview
This project demonstrates an end-to-end Machine Learning Operations (MLOps) pipeline for a Hotel Recommendation System. It covers the entire machine learning lifecycle: from data extraction and preprocessing, model training with experiment tracking, to deploying the model as a fully containerized REST API microservice.

This architecture is designed to reflect industry standards in travel-technology companies, focusing on scalability, reproducibility, and seamless deployment.

## Tech Stack & Architecture
* **Data Processing:** `Pandas`, `Scikit-learn` (Label Encoding)
* **Model Training:** `RandomForestRegressor`
* **Experiment Tracking & Model Registry:** `MLflow`
* **Model Serving (API):** `FastAPI`, `Uvicorn`
* **Containerization:** `Docker`

## 📂 Project Structure
```text
hotel_mlops_project/
├── api/
│   ├── model/               # Exported MLflow model artifacts & Encoders
│   ├── main.py              # FastAPI application
│   ├── Dockerfile           # Docker configuration for API serving
│   └── requirements.txt     # API dependencies
├── data/                    # Raw and processed datasets
├── src/
│   ├── preprocess.py        # Data cleaning and feature encoding scripts
│   └── train.py             # Model training and MLflow logging scripts
├── mlruns/                  # MLflow tracking local directory
└── requirements.txt         # Local development dependencies
```
## How to Run Locally
1. Data Processing and Training
Ensure you have the virtual environment activated, then run:

```Bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
```
2. Run API via Docker (Production Simulation)
Navigate to the api directory and build the Docker image:

```Bash
cd api
docker build -t hotel-recommendation-api .
docker run -p 8000:8000 hotel-recommendation-api
```
## API Usage
Once the Docker container is running, the API documentation (Swagger UI) can be accessed at:
http://localhost:8000/docs

Example Request:
POST /predict

```JSON
{
  "user_id": 5,
  "hotel_id": 12
}
```
Example Response:

```JSON
{
  "user_id": 5,
  "hotel_id": 12,
  "predicted_rating": 4.16
}
```
