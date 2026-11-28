"""Stage 4: drift (PSI), per-group MAE, retrain v2, promotion rule, holdout predictions.
Run from the repo root:  py -3.12 -m src.mp115.monitor
"""
import os
import re
import shutil

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.mp115.cleaning import load_params
from src.mp115.modeling import build_model, file_sha256, mae

FINDINGS = "reports/findings.yaml"
IMPROVEMENT_NEEDED = 0.05      # client rule: v2 must be at least 5% better


def psi(expected, actual, bins=10):
    """Population Stability Index: train (expected) vs later (actual), 10 bins."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def set_value(text, key, value):
    """Replace `key: null` inside the monitoring block of findings.yaml."""
    head, sep, tail = text.partition("monitoring:")
    pattern = rf"(^\s*{re.escape(key)}:\s*)[^#\n]*"
    tail, n = re.subn(pattern, lambda m: f"{m.group(1)}{value}  ", tail, count=1, flags=re.M)
    if n == 0:
        raise ValueError(f"key not found in findings.yaml: {key}")
    return head + sep + tail


def main():
    params = load_params()
    paths, m = params["paths"], params["model"]
    target = m["target"]
    num, cat = m["numeric_features"], m["categorical_features"]
    features = num + cat
    name = m["registered_model_name"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(m["experiment_name"])
    client = MlflowClient()
    if len(client.search_model_versions(f"name='{name}'")) != 1:
        raise SystemExit("Expected exactly 1 registered version (v1). Delete mlflow.db and "
                         "run  py -3.12 -m src.mp115.train  first.")

    train = pd.read_csv(paths["train_clean"])
    later = pd.read_csv(paths["later_raw"]).sort_values("date").reset_index(drop=True)
    holdout = pd.read_csv(paths["holdout"])

    # 1. PSI for every numeric column
    psi_values = {c: round(psi(train[c], later[c]), 4) for c in num}
    most_drifted = max(psi_values, key=psi_values.get)
    print("PSI:", psi_values, "-> most drifted:", most_drifted)

    # 2. v1 MAE on later.csv per hostel
    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")
    later["pred_v1"] = v1.predict(later[features])
    mae_group = {h: round(mae(g[target], g["pred_v1"]), 4) for h, g in later.groupby("hostel")}
    worst = max(mae_group, key=mae_group.get)
    print("MAE v1 by hostel:", mae_group, "-> worst:", worst)

    # 3. retrain v2 on train + first 70% of later, compare on last 30%
    cut = int(len(later) * 0.7)
    later_fit, check = later.iloc[:cut], later.iloc[cut:]
    fit = pd.concat([train, later_fit[train.columns]], ignore_index=True)

    best_run = mlflow.search_runs(experiment_names=[m["experiment_name"]],
                                  order_by=["metrics.mae_valid ASC"]).iloc[0]
    run_cfg = next(r for r in m["runs"] if r["name"] == best_run["tags.mlflow.runName"])

    v2 = build_model(run_cfg, m).fit(fit[features], fit[target])
    mae_v1 = mae(check[target], check["pred_v1"])
    mae_v2 = mae(check[target], v2.predict(check[features]))
    promote = mae_v2 <= (1 - IMPROVEMENT_NEEDED) * mae_v1
    decision = "promote" if promote else "keep"
    print(f"MAE on last 30%: v1={mae_v1:.4f} v2={mae_v2:.4f} -> {decision}")

    with mlflow.start_run(run_name="v2_retrain"):
        mlflow.log_param("data_sha256", file_sha256(paths["train_raw"]))
        mlflow.log_params({k: v for k, v in run_cfg.items() if k != "name"})
        mlflow.log_param("trained_on", "train_clean + first 70% of later")
        mlflow.log_metric("mae_check_v1", mae_v1)
        mlflow.log_metric("mae_check_v2", mae_v2)
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
    mv2 = mlflow.register_model(info.model_uri, name)
    champion_version = int(mv2.version) if promote else 1
    client.set_registered_model_alias(name, "champion", champion_version)
    champion_model = v2 if promote else v1
    print("champion is now version", champion_version)

    # 4. holdout predictions from the champion
    os.makedirs("predictions", exist_ok=True)
    out = pd.DataFrame({"record_id": holdout["record_id"],
                        "prediction": champion_model.predict(holdout[features])})
    out.to_csv("predictions/holdout_predictions.csv", index=False)
    joblib.dump(champion_model, m["model_path"])      # the API serves the champion
    print("wrote predictions/holdout_predictions.csv with", len(out), "rows")

    # 5. fill the monitoring block of findings.yaml
    text = open(FINDINGS, encoding="utf-8").read()
    for c, v in psi_values.items():
        text = set_value(text, c, v)
    text = set_value(text, "most_drifted_feature", most_drifted)
    for h, v in mae_group.items():
        text = set_value(text, f"'{h}'", v)
    text = set_value(text, "worst_served_group", f"'{worst}'")
    text = set_value(text, "mae_v1_on_check", round(mae_v1, 4))
    text = set_value(text, "mae_v2_on_check", round(mae_v2, 4))
    text = set_value(text, "decision", decision)
    open(FINDINGS, "w", encoding="utf-8").write(text)
    print("findings.yaml monitoring block filled")


if __name__ == "__main__":
    main()
