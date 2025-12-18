# pipelines/training_pipeline.py
from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=True)
def train_pipeline(data_path: str, model_name: str = "Logistic Regression"):
    df = ingest_data(data_path)
    X_train_scaled, X_test_scaled, y_train, y_test = clean_data(df)
    
    # Create config and pass to train_model
    config = ModelNameConfig(model_name=model_name)
    model = train_model(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        model_name_config=config
    )
    
    metrics = evaluate_model(model, X_test_scaled, y_test)