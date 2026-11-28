"""Copy the MLflow details into reports/findings.yaml. Run: py -3.12 fill_tracking.py"""
import re
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
NAME = "mp115-waste-model"
runs = mlflow.search_runs(experiment_names=["mp115-waste"], order_by=["metrics.mae_valid ASC"])
best = runs.iloc[0]["run_id"]
versions = MlflowClient().search_model_versions(f"name='{NAME}'")
champ = MlflowClient().get_model_version_by_alias(NAME, "champion").version

path = "reports/findings.yaml"
text = open(path, encoding="utf-8").read()
text = re.sub(r"(experiment_name:\s*).*", r'\1"mp115-waste"', text, count=1)
text = re.sub(r"(best_run_id:\s*)[^#\n]*", rf'\1"{best}"  ', text, count=1)
text = re.sub(r"(registered_model_name:\s*).*", rf'\1"{NAME}"', text, count=1)
text = re.sub(r"(champion_version:\s*).*", rf"\g<1>{champ}", text, count=1)
open(path, "w", encoding="utf-8").write(text)
print("runs:", len(runs), "| versions:", len(versions), "| best:", best, "| champion:", champ)
