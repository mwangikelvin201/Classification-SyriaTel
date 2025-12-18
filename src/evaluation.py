import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluation(ABC):
    """
    Abstract base class for model evaluation strategies.
    """

    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class Accuracy(Evaluation):
    """
    Accuracy evaluation strategy.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy calculated: {accuracy}")
            return accuracy
        except Exception as e:
            logging.error(f"Error calculating accuracy: {e}")
            raise e
        

class Precision(Evaluation):
    """
    Precision evaluation strategy that uses precision score metric.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            precision = precision_score(y_true, y_pred)
            logging.info(f"Precision calculated: {precision}")
            return precision
        except Exception as e:
            logging.error(f"Error calculating precision: {e}")
            raise e


class Recall(Evaluation):
    """
    Recall evaluation strategy that uses recall score metric.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            recall = recall_score(y_true, y_pred)
            logging.info(f"Recall calculated: {recall}")
            return recall
        except Exception as e:
            logging.error(f"Error calculating recall: {e}")
            raise e


class F1(Evaluation):
    """
    F1 score evaluation strategy that uses F1 score metric.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            f1 = f1_score(y_true, y_pred)
            logging.info(f"F1 score calculated: {f1}")
            return f1
        except Exception as e:
            logging.error(f"Error calculating F1 score: {e}")
            raise e