# run_pipeline.py
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
import mlflow

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    data_path = "/home/kamwas/datascience/Classification-SyriaTel/data/bigml_59c28831336c6604c800002a.csv"
    
    models = ["Logistic Regression", "Random Forest", "Random Forest Pipeline"]
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Training and Evaluating: {model_name}")
        print(f"{'='*60}\n")
        
        train_pipeline(data_path=data_path, model_name=model_name)

