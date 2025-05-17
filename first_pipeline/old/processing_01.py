import pandas as pd

df = pd.read_csv("feature_engineered_data.csv")      # jouw samengestelde set
df.sort_values(["subject_id", "stay_id", "hfnc_episode", "hour_ts"],
               inplace=True)

vitals      = ["spo2", "resp_rate", "heart_rate", "temperature",
               "sbp", "dbp", "rox", "sf_ratio", "fio2_frac"]
labs        = ["paco2", "ph", "pao2"]        # neem wat er >70 % gevuld is
context_num = ["hrs_since_ep_start", "episode_len"]
num_cols    = vitals + labs + context_num

cat_cols  = ["rox_class", "humidification"]
df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)


#Binning

# Flow-bins (L/min)
flow_edges = [0, 20, 40, 60]
# FiO₂-bins (%)
fio2_edges = [21, 40, 60, 80, 100]

df["flow_bin"] = pd.cut(df.o2_flow,  flow_edges, labels=False, right=False)
df["fio2_bin"] = pd.cut(df.fio2,     fio2_edges, labels=False, right=False)

n_flow  = len(flow_edges) - 1
n_fio   = len(fio2_edges) - 1
df["action"] = df["flow_bin"] * n_fio + df["fio2_bin"]
n_actions    = df["action"].max() + 1

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # Display the first 10 rows of the dataframe
    print("First 10 rows of the DataFrame:")
    print(df.head(10))

    # Save the dataframe to a new CSV file
    df.to_csv('feature_engineered_data.csv', index=False)

    print("DataFrame saved to feature_engineered_data.csv")
