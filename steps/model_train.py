import logging
import pandas as pd
from zenml import step
from src.model_dev import RandomForestModel, RandomForestPipeline, LogisticRegressionModel
from steps.config import ModelNameConfig
@step
def train_model(
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name_config: ModelNameConfig
):
    """Trains a machine learning model based on the specified configuration.

    Args:
        X_train_scaled (pd.DataFrame): The scaled training features.
        X_test_scaled (pd.DataFrame): The scaled testing features.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The testing labels.
        model_name_config (ModelNameConfig): Configuration specifying the model type.

    Returns:
        ClassifierMixin: The trained machine learning model.
    """
    try:
        model = None
        trained_model = None

        if model_name_config.model_name == "Random Forest":
            model = RandomForestModel()
            trained_model = model.train(X_train_scaled, y_train)
            logging.info("Random Forest model trained successfully.")
        elif model_name_config.model_name == "Logistic Regression":
            model = LogisticRegressionModel()
            trained_model = model.train(X_train_scaled, y_train)
            logging.info("Logistic Regression model trained successfully.")
        elif model_name_config.model_name == "Random Forest Pipeline":
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