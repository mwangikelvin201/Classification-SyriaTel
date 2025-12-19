import logging
import pandas as pd
from zenml import step
from src.evaluation import Accuracy, Precision, Recall, F1
import mlflow
from zenml.client import Client

# Get the tracking URI from ZenML's existing configuration
client = Client()
experiment_tracker = client.active_stack.experiment_tracker
tracking_uri = experiment_tracker.config.tracking_uri
mlflow.set_tracking_uri(tracking_uri)

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Evaluates the trained model using various evaluation metrics."""
    
    # Don't manually start a run - ZenML handles this
    try:
        predictions = model.predict(X_test_scaled)
        
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        
        precision_class = Precision()
        precision = precision_class.calculate(y_test, predictions)
        mlflow.log_metric("precision", precision)
        
        recall_class = Recall()
        recall = recall_class.calculate(y_test, predictions)
        mlflow.log_metric("recall", recall)
        
        f1_class = F1()
        f1 = f1_class.calculate(y_test, predictions)
        mlflow.log_metric("f1", f1)
        
        logging.info(f"Model evaluated - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }