"""Model building and metrics for MP115."""
import hashlib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_model(run, m):
    """Build a pipeline from one entry of model.runs in params.yaml."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), m["numeric_features"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), m["categorical_features"]),
    ])
    kind = run["type"]
    if kind == "ridge":
        reg = Ridge(alpha=run["alpha"])
    elif kind == "random_forest":
        reg = RandomForestRegressor(n_estimators=run["n_estimators"],
                                    max_depth=run["max_depth"], random_state=42)
    elif kind == "gradient_boosting":
        reg = GradientBoostingRegressor(n_estimators=run["n_estimators"],
                                        learning_rate=run["learning_rate"],
                                        max_depth=run["max_depth"], random_state=42)
    else:
        raise ValueError(f"unknown model type: {kind}")
    return Pipeline([("pre", pre), ("reg", reg)])


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))
