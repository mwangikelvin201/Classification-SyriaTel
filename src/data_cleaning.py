import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import logging
import scipy.stats as stats
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataStrategy(ABC):
    """
    Abstract base class for data cleaning strategies.
    """

    @abstractmethod
    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing for cleaning the dataset
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(columns = ['phone number','account length','area code','state'])

            variables_to_encode = ['international plan','voice mail plan']
            data = pd.get_dummies(data,columns = variables_to_encode,drop_first = True,dtype = int)

            data['churn'] = data['churn'].astype(int)
            """
            Args:
            - df: The input DataFrame

            Correlation Analysis to remove highly correlated features
            """

            df_corr = data.drop(columns = 'churn')
            corr_matrix = df_corr.corr()

            # Set threshold for correlation
            threshold = 0.8

            corr_pairs = corr_matrix.unstack().sort_values(kind="heap", ascending=False)
            high_corr_pairs = [(i, j) for i, j in corr_pairs.index if i != j and abs(corr_pairs[i, j]) > threshold]

            # Drop one of the variables from each pair
            for i, j in high_corr_pairs:
                if j in data.columns:
                    data.drop(j, axis=1, inplace=True)

            """Outlier Detection and Removal using Z-Score Method"""
            # Calculate z-scores for each column
            z_scores = np.abs(zscore(data))

            # Define a threshold for outliers
            threshold = 4

            # Create a boolean mask indicating outliers
            outlier_mask = (z_scores > threshold).any(axis=1)

            # Remove outliers from the dataset
            data = data[~outlier_mask]

            return data
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise e
        

class DataDivideStrategy(DataStrategy):
    """
    Data division strategy for splitting the dataset into training and testing sets.
    """

    def handle_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = df.drop('churn', axis=1)
            y = df['churn']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()

            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )

            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logging.error(f"Error during data division: {e}")
            raise e


class DataCleaning:
    """
    Data cleaning context class that uses different strategies for cleaning data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        '''Initialize the DataCleaning with a specific strategy.'''
        self.data = data
        self._strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        '''Handle data using the specified strategy.'''
        try:
            return self._strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e