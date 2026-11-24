import pandas as pd

TRAIN = "data/raw/train.csv"
HOLDOUT = "data/raw/holdout_features.csv"
LATER = "data/raw/later.csv"
TARGET = "waste_kg"
GROUP = "hostel"
NUM = ["students_registered", "attendance_pct", "menu_rating",
       "temp_c", "cooked_kg", "plate_size_cm"]
TEXT = ["meal", "hostel"]

df = pd.read_csv(TRAIN)
hold = pd.read_csv(HOLDOUT)
later = pd.read_csv(LATER)

print("1. LEAKING COLUMN")
extra = [c for c in df.columns if c not in hold.columns and c != TARGET]
print("   columns in train but not in holdout (excluding target):", extra)
print("   correlation with target:")
print(df[NUM].corrwith(df[TARGET]).round(3).to_string())

print("\n2. UNIT MIXUP (group median / overall median)")
ratio = df.groupby(GROUP)[NUM].median() / df[NUM].median()
print(ratio.round(2).to_string())

print("\n3. SENTINEL VALUES (most repeated value per column)")
for c in NUM:
    vc = df[c].value_counts()
    print(f"   {c}: value={vc.index[0]}, count={vc.iloc[0]}")
print("   min/max per column:")
print(df[NUM].agg(["min", "max"]).T.to_string())

print("\n4. DUPLICATE ROWS")
print("   extra copies:", df.duplicated().sum())

print("\n5. STUCK SENSOR (longest run of identical values, rows sorted by date)")
d = df.drop_duplicates().sort_values(["date", "record_id"]).reset_index(drop=True)
for c in NUM:
    run_id = (d[c] != d[c].shift()).cumsum()
    sizes = d.groupby(run_id)[c].transform("size")
    i = sizes.idxmax()
    grp = d[run_id == run_id[i]]
    print(f"   {c}: longest run={sizes.max()} rows, value={d.loc[i, c]}, "
          f"{grp['date'].iloc[0]} to {grp['date'].iloc[-1]}")

print("\n6. TARGET TYPOS (largest jumps between sorted targets)")
s = df[TARGET].sort_values().reset_index(drop=True)
gaps = (s.shift(-1) / s).dropna()
print("   biggest ratio between neighbours:", round(gaps.max(), 2),
      "at value", s[gaps.idxmax()])
print("   top 15 values:", s.tail(15).to_list())

print("\n7. SPELLING VARIANTS")
for c in TEXT:
    print(f"   {c} in train:", sorted(df[c].dropna().unique()))
    print(f"   {c} in later (clean):", sorted(later[c].dropna().unique()))
