# 1. FIRST: Add path fix before ANY imports
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple
import sys
import os
    
# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train_scaled"],
    Annotated[pd.DataFrame, "X_test_scaled"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:

    """Cleans the input data using defined strategies and returns training and testing sets."""
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train_scaled, X_test_scaled, y_train, y_test = data_cleaning.handle_data()

        logging.info(f"Data cleaned: {X_train_scaled.shape[0]} training rows, {X_test_scaled.shape[0]} testing rows")
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise e