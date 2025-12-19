import json
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel  # Changed from BaseParameters
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.data_ingestion import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig  # Import your ModelNameConfig
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseModel):  # Changed from BaseParameters
    """Configuration for deployment trigger."""
    min_accuracy: float = 0.92


@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Decides whether to deploy the model based on accuracy."""
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=False, settings={"docker": docker_settings})  # Changed enable_cache to False and fixed settings key
def continuous_deployment_pipeline(
    data_path: str,
    model_name: str = "Random Forest",
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """A continuous deployment pipeline that trains, evaluates, and deploys a model if it meets accuracy criteria."""
    
    # Ingest data
    df = ingest_data(data_path=data_path)
    X_train_scaled, X_test_scaled, y_train, y_test = clean_data(df)
    
    # Create config and pass to train_model
    model_config = ModelNameConfig(model_name=model_name)
    model = train_model(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        model_name_config=model_config
    )
    
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Create deployment trigger config
    deployment_config = DeploymentTriggerConfig(min_accuracy=min_accuracy)
    deployment_decision = deployment_trigger(
        accuracy=metrics["accuracy"],
        config=deployment_config
    )
    
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,  # Note: parameter name might be deploy_decision
        workers=workers,
        timeout=timeout,
    )