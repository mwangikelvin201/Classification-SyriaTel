from pydantic import BaseModel
from typing import Literal

class ModelNameConfig(BaseModel):
    model_name: Literal[
        "Logistic Regression",
        "Random Forest",
        "Random Forest Pipeline"
    ] = "Logistic Regression"