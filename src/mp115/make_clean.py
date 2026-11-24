"""Run:  python -m src.mp115.make_clean   (from the repo root)"""
import os
import pandas as pd
from src.mp115.cleaning import load_params, clean


def main():
    params = load_params()
    df = pd.read_csv(params["paths"]["train_raw"])
    out = clean(df, params["cleaning"])
    path = params["paths"]["train_clean"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    print("rows before:", len(df), "rows after:", len(out))
    print(out[["plate_size_cm", "waste_kg", "attendance_pct"]].describe().T)


if __name__ == "__main__":
    main()
