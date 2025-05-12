import pandas as pd

# Load the preprocessed data
df = pd.read_csv('preprocessed_data_reduced_4.csv')

import pandas as pd

# --- load & basic hygiene --------------------------------------------------
df["hour_ts"] = pd.to_datetime(df["hour_ts"])
df = df.sort_values(["subject_id", "stay_id", "hfnc_episode", "hour_ts"])

# (optional) if `hfnc_episode` came in as float -> int
df["hfnc_episode"] = df["hfnc_episode"].astype("int16")

# --- compute hours since the start of *each* episode -----------------------
df["ep_start_ts"] = df.groupby(
        ["subject_id", "stay_id", "hfnc_episode"]
    )["hour_ts"].transform("min")

df["hrs_since_ep_start"] = (
    df["hour_ts"].sub(df["ep_start_ts"]).dt.total_seconds() / 3600
)

# --- ROX index -------------------------------------------------------------
df["fio2_frac"] = df["fio2"] / 100          # 50 → 0.50
df["rox"] = (df["spo2"] / df["fio2_frac"]) / df["resp_rate"]

# --- ROX-risk class (classic thresholds) -----------------------------------
def rox_class(h, r):
    if h < 2:           return None
    if 2 <= h < 6:      return "high" if r < 2.85 else "low" if r >= 4.88 else "grey"
    if 6 <= h < 12:     return "high" if r < 3.47 else "low" if r >= 4.88 else "grey"
    return               "high" if r <= 3.85 else "low" if r >= 4.88 else "grey"

df["rox_class"] = df.apply(
    lambda row: rox_class(row["hrs_since_ep_start"], row["rox"]), axis=1
)

df["episode_len"] = df.groupby(
    ["subject_id", "stay_id", "hfnc_episode"]
).transform("size")

df["sf_ratio"] = df["spo2"] / df["fio2_frac"]          # SpO2 uses %, FiO2 fraction
df["pf_ratio"] = df["pao2"] / df["fio2_frac"]        # PaO2 uses mmHg, FiO2 fraction

df["mask_wean"] = (
    (df["rox_class"] == "high") |                       # ROX-high risk
    (df["spo2"] < 90) |                                 # or low saturation
    (df["sf_ratio"] < 235)                              # or S F equivalent to P F < 200
)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # Display the first 10 rows of the dataframe
    print("First 10 rows of the DataFrame:")
    print(df.head(10))

    # Save the dataframe to a new CSV file
    df.to_csv('feature_engineered_data.csv', index=False)

    print("DataFrame saved to feature_engineered_data.csv")


# Display basic information about the dataframe
print("DataFrame Info:")
print(df.info())