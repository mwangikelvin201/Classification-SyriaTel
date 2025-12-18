import logging
import pandas as pd
from zenml import step
from src.evaluation import Accuracy, Precision, Recall, F1

@step
def evaluate_model(
    model,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Evaluates the trained model using various evaluation metrics."""
    try:
        predictions = model.predict(X_test_scaled)
        
        accuracy_class = Accuracy()
        accuracy = accuracy_class.calculate(y_test, predictions)
        
        precision_class = Precision()
        precision = precision_class.calculate(y_test, predictions)
        
        recall_class = Recall()
        recall = recall_class.calculate(y_test, predictions)
        
        f1_class = F1()
        f1 = f1_class.calculate(y_test, predictions)
        
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }