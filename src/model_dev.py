import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class Model(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def train(self, X_train_scaled, y_train):
        """
        Trains the model with the provided training data.
        Args:
            X_train_scaled (pd.DataFrame): The scaled training features.
            y_train (pd.Series): The training labels.
        """
        pass

class LogisticRegressionModel(Model):
    def train(self, X_train_scaled, y_train, **kwargs):
        try:
            model = LogisticRegression(**kwargs)
            model.fit(X_train_scaled, y_train)
            logging.info("Logistic Regression model trained.")
            return model
        except Exception as e:
            logging.error(f"Error training Logistic Regression model: {e}")
            raise e
       

class RandomForestModel(Model):
    def train(self, X_train_scaled, y_train, **kwargs):
        try:
            model = RandomForestClassifier(**kwargs)
            model.fit(X_train_scaled, y_train)
            logging.info("Random Forest model trained.")
            return model
        except Exception as e:
            logging.error(f"Error training Random Forest model: {e}")
            raise e
        
class RandomForestPipeline(Model):
    def train(self, X_train_scaled, y_train, **kwargs):
        try:
            pipe = Pipeline([
                ('forest', RandomForestClassifier(random_state=123))
            ])

            # Define the parameter grid
            grid = {
                'forest__max_depth': [None, 2, 6, 10],
                'forest__min_samples_split': [5, 10]
            }

            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=pipe,
                                    param_grid=grid,
                                    scoring='accuracy',
                                    cv=5)

            # Fit the grid search
            grid_search.fit(X_train_scaled, y_train)

            logging.info("Random Forest Pipeline model trained.")
            return grid_search
        except Exception as e:
            logging.error(f"Error training Random Forest Pipeline model: {e}")
            raise e