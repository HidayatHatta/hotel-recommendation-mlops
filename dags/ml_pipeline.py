from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'hotel_recommendation_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Task 1: Data Preprocessing
    # Ganti path sesuai dengan direktori absolut proyek Anda
    preprocess_task = BashOperator(
        task_id='preprocess_data',
        bash_command='python /path/to/hotel_mlops_project/src/preprocess.py' 
    )

    # Task 2: Model Training
    train_task = BashOperator(
        task_id='train_model',
        bash_command='python /path/to/hotel_mlops_project/src/train.py'
    )

    # Definisi urutan eksekusi (DAG topology)
    preprocess_task >> train_task