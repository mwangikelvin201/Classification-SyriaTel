import logging
import pandas as pd
from zenml import step
from src.model_dev import RandomForestModel, RandomForestPipeline, LogisticRegressionModel
from steps.config import ModelNameConfig
import mlflow
from zenml.client import Client

# Get the tracking URI from ZenML's existing configuration
client = Client()
experiment_tracker = client.active_stack.experiment_tracker
tracking_uri = experiment_tracker.config.tracking_uri
mlflow.set_tracking_uri(tracking_uri)

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name_config: ModelNameConfig
):
    """Trains a machine learning model based on the specified configuration."""
    
    # Remove autolog since ZenML handles MLflow integration
    # mlflow.sklearn.autolog()
    
    # Don't manually start a run - ZenML handles this with the decorator
    try:
        trained_model = None
        if model_name_config.model_name == "Random Forest":
            params = {
                "model_name": "Random Forest",
                "n_estimators": 1000,
                "max_depth": 10,
                "min_samples_split": 2,
                "random_state": 84
            }
            mlflow.log_params(params)
            
            model = RandomForestModel()
            trained_model = model.train(X_train_scaled, y_train)
            logging.info("Random Forest model trained successfully.")
            
        elif model_name_config.model_name == "Logistic Regression":
            params = {
                "model_name": "Logistic Regression",
                "max_iter": 10000,
                "random_state": 84,
                "solver": "lbfgs"
            }
            mlflow.log_params(params)
            
            model = LogisticRegressionModel()
            trained_model = model.train(X_train_scaled, y_train)
            logging.info("Logistic Regression model trained successfully.")
            
        elif model_name_config.model_name == "Random Forest Pipeline":
            params = {
                "model_name": "Random Forest Pipeline",
                "n_estimators": 120,
                "max_depth": 10,
                "random_state": 84
            }
            mlflow.log_params(params)
            
            model = RandomForestPipeline()
            trained_model = model.train(X_train_scaled, y_train)
            logging.info("Random Forest Pipeline model trained successfully.")
            
        else:
            logging.error(f"Unsupported model name: {model_name_config.model_name}")
            raise ValueError(f"Unsupported model name: {model_name_config.model_name}")
        
        if trained_model is None:
            raise ValueError("Model training returned None. Check your model's train() method.")
        
        return trained_model
        
    except Exception as e:
        logging.error(f"Error occurred while training model: {e}")
        raise