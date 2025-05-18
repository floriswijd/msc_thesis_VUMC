import pandas as pd
import yaml
from pathlib import Path

RAW_CSV  = Path("data/feature_engineered_data.csv")
OUT_PARQ = Path("data/hfnc_episodes.parquet")

flow_edges = [0, 20, 40, 71]
fio2_edges = [21, 40, 60, 80, 101]

def main(debug=False):
    df = pd.read_csv(RAW_CSV)
    df.sort_values(["subject_id","stay_id","hfnc_episode","hour_ts"], inplace=True)
    vitals = ["spo2","resp_rate","heart_rate","temperature","sbp","dbp","rox","sf_ratio","fio2_frac"]
    labs   = ["paco2","ph","pao2"]
    ctx    = ["hrs_since_ep_start","episode_len"]
    num_cols = vitals + labs + ctx
    df["flow_bin"] = pd.cut(df.o2_flow, flow_edges, labels=False, right=False)
    df["fio2_bin"] = pd.cut(df.fio2,  fio2_edges, labels=False, right=False)

    if debug:
        if df["flow_bin"].isnull().any():
            print(f"Warning: {df['flow_bin'].isnull().sum()} NaNs found in 'flow_bin' after binning. Check o2_flow values and flow_edges.")
        if df["fio2_bin"].isnull().any():
            print(f"Warning: {df['fio2_bin'].isnull().sum()} NaNs found in 'fio2_bin' after binning. Check fio2 values and fio2_edges.")

    n_actions = (len(flow_edges)-1)*(len(fio2_edges)-1)
    df["action"]  = df["flow_bin"]*(len(fio2_edges)-1) + df["fio2_bin"]

    if debug and df["action"].isnull().any():
        print(f"Warning: {df['action'].isnull().sum()} NaN values found in 'action' column BEFORE astype conversion.")
        print("This indicates that some o2_flow or fio2 values might still be outside the adjusted bin ranges.")
        print("Problematic rows (o2_flow, fio2 where action is NaN):")
        print(df[df["action"].isnull()][['o2_flow', 'fio2', 'flow_bin', 'fio2_bin']].head())
    df["reward"] = df["spo2"].apply(lambda x: 0.1 if 92<=x<=96 else -0.05)
    terminal_idx = df.groupby(["subject_id","stay_id","hfnc_episode"]).tail(1).index
    outcome_val  = {"Success":1.0,"InvasiveVent":-1,"Intubation":-1.0,"Death":-1.0}
    df.loc[terminal_idx,"reward"] = df.loc[terminal_idx,"outcome_label"].map(outcome_val)
    df["done"] = False
    df.loc[terminal_idx,"done"] = True
    df[num_cols] = df[num_cols].astype("float32")
    df["action"] = df["action"].astype("int64")
    df["reward"] = df["reward"].astype("float32")
    df["done"]   = df["done"].astype("bool")

    if debug:
        print("\\nNaN counts per column before saving:")
        print(df.isnull().sum())
        print(f"\\nTotal NaN count in DataFrame: {df.isnull().sum().sum()}")
    OUT_PARQ.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQ, compression="snappy", index=False)
    if debug:
        print(df.head())
        print(f"Saved {len(df):,} rows  •  n_actions={n_actions}")
    cfg = dict(flow_edges=flow_edges, fio2_edges=fio2_edges, n_actions=int(n_actions))
    with open("config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

if __name__ == "__main__":
    main(debug=True)
