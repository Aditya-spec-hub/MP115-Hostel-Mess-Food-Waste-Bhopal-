import pandas as pd

TRAIN = "data/raw/train.csv"
HOLDOUT = "data/raw/holdout_features.csv"

df = pd.read_csv(TRAIN)
hold = pd.read_csv(HOLDOUT)

print("Shape:", df.shape)
print("\nColumns only in train:", set(df.columns) - set(hold.columns))

print("\n--- Duplicate rows (extra copies):", df.duplicated().sum())

num_cols = ["students_registered", "attendance_pct", "menu_rating",
            "temp_c", "cooked_kg", "plate_size_cm", "waste_kg"]
print("\n--- Describe ---")
print(df[num_cols].describe().T)

print("\n--- Per hostel: mean of each numeric column ---")
print(df.groupby("hostel")[num_cols].mean().T)

print("\n--- Most common values per numeric column ---")
for c in num_cols:
    print(c, df[c].value_counts().head(3).to_dict())

print("\n--- Text columns ---")
print(df["meal"].value_counts(dropna=False))
print(df["hostel"].value_counts(dropna=False))

print("\n--- Largest waste_kg values ---")
print(df["waste_kg"].sort_values(ascending=False).head(15).to_list())
