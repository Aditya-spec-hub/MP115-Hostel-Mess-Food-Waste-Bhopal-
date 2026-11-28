"""Train v1 with MLflow tracking.  Run from the repo root:
    python -m src.mp115.train
"""
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from src.mp115.cleaning import load_params
from src.mp115.modeling import build_model, file_sha256, mae


def main():
    params = load_params()
    paths, m = params["paths"], params["model"]
    features = m["numeric_features"] + m["categorical_features"]

    train = pd.read_csv(paths["train_clean"]).sort_values("date")
    later = pd.read_csv(paths["later_raw"])
    data_hash = file_sha256(paths["train_raw"])  # hash of the ORIGINAL train.csv

    cut = int(len(train) * (1 - m["valid_fraction"]))
    tr, va = train.iloc[:cut], train.iloc[cut:]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(m["experiment_name"])

    results = []
    for run in m["runs"]:
        with mlflow.start_run(run_name=run["name"]) as active:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params({k: v for k, v in run.items() if k != "name"})

            # validation: fit on the early part of train, test on the latest part
            model = build_model(run, m).fit(tr[features], tr[m["target"]])
            mae_valid = mae(va[m["target"]], model.predict(va[features]))

            # final v1 model: fit on all cleaned train, test on later.csv
            model = build_model(run, m).fit(train[features], train[m["target"]])
            mae_later = mae(later[m["target"]], model.predict(later[features]))

            mlflow.log_metric("mae_valid", mae_valid)
            mlflow.log_metric("mae_later", mae_later)
            mlflow.sklearn.log_model(model, name="model",
                                     serialization_format="cloudpickle")
            results.append((mae_valid, run["name"], active.info.run_id, model, mae_later))
            print(f"{run['name']:18s} mae_valid={mae_valid:.3f} mae_later={mae_later:.3f}")

    # best run = lowest validation MAE (not chosen using later.csv)
    best = min(results, key=lambda r: r[0])
    _, name, run_id, model, _ = best
    print("best:", name, "run_id:", run_id)

    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", m["registered_model_name"])
    client.set_registered_model_alias(m["registered_model_name"], "champion", mv.version)

    os.makedirs(os.path.dirname(m["model_path"]), exist_ok=True)
    joblib.dump(model, m["model_path"])
    print(json.dumps({"experiment_name": m["experiment_name"], "best_run_id": run_id,
                      "registered_model_name": m["registered_model_name"],
                      "champion_version": int(mv.version)}, indent=2))


if __name__ == "__main__":
    main()
