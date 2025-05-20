import pandas as pd
import yaml
from pathlib import Path

TARGET_SPO2 = 94           # centre of desired range
SETPOINT_K  = 8            # ±k → reward falls to 0
FLOW_COST   = 0.002        # per L/min
FIO2_COST   = 0.005        # per % (use 0.5 % if you prefer frac)

RAW_CSV  = Path("data/feature_engineered_data.csv")
OUT_PARQ = Path("data/hfnc_episodes.parquet")

flow_edges = [0, 20, 40, 71]
fio2_edges = [21, 40, 60, 80, 101]

def main(debug=False):
    df = pd.read_csv(RAW_CSV)
    df.sort_values(["subject_id","stay_id","hfnc_episode","hour_ts"], inplace=True)
    df = add_episode_outcome(df)

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
    
    terminal_idx = df.groupby(["subject_id","stay_id","hfnc_episode"]).tail(1).index
    df["done"] = False
    df.loc[terminal_idx,"done"] = True
    df = add_rewards(df)
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
    # Save to CSV
    csv_path = OUT_PARQ.parent / "hfnc_episodes_01.csv"
    df.to_csv(csv_path, index=False)

    if debug:
        print(df.head())
        print(f"Saved {len(df):,} rows to {OUT_PARQ} and {csv_path} •  n_actions={n_actions}")
    cfg = dict(flow_edges=flow_edges, fio2_edges=fio2_edges, n_actions=int(n_actions))
    with open("config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


def add_rewards(df: pd.DataFrame) -> pd.DataFrame:
    # ------------------------------------------------------------------
    # 1. sort by subject/stay/episode/time so shift() works as expected
    # ------------------------------------------------------------------
    df = df.sort_values(
        ["subject_id", "stay_id", "hfnc_episode", "hour_ts"]
    ).reset_index(drop=True)

        # 2. Next-hour SpO2 inside each episode
    df["spo2_next"] = (
        df.groupby(["subject_id", "stay_id", "hfnc_episode"])["spo2"]
        .shift(-1)
    )

    # ▶︎ NEW: for terminal step, pretend spo2_next == spo2  ◀︎
    df["spo2_next"].fillna(df["spo2"], inplace=True)      # <── moved up

    # 3. Component A – set-point reward  (now no NaNs)
    df["r_A"] = 1 - (df["spo2_next"] - TARGET_SPO2).abs() / SETPOINT_K
    df["r_A"] = df["r_A"].clip(-1, 1)

    # 4. Component B – Δ-SpO₂
    df["r_B"] = (df["spo2_next"] - df["spo2"]) / 4.0
    df["r_B"] = df["r_B"].clip(-1, 1)

    # ------------------------------------------------------------------
    # 5. Component C – cost of the current setting
    # ------------------------------------------------------------------
    df["r_C"] = -(FIO2_COST * df["fio2"] + FLOW_COST * df["o2_flow"])

    # ------------------------------------------------------------------
    # 6. Component D – wean / escalation signal
    # ------------------------------------------------------------------
    df["r_D"] = 0.0
    wean_ok = (df["mask_wean"]) & (df["spo2_next"] >= 92)
    df.loc[wean_ok, "r_D"] = +0.3

    escalate = df["ventilation_to"].isin(["NIV", "InvasiveVent"])
    df.loc[escalate, "r_D"] = -0.3

    # ------------------------------------------------------------------
    # 7. Sum step rewards with chosen weights
    # ------------------------------------------------------------------
    df["reward"] = (
        0.4 * df["r_A"]
      + 0.2 * df["r_B"]
      + 0.2 * df["r_C"]
      + 0.1 * df["r_D"]
    )

    # ------------------------------------------------------------------
    # 8. Add terminal outcome reward
    # ------------------------------------------------------------------
    outcome_map = {
        "Success":      +1.0,
        "NIV":          -0.5,
        "InvasiveVent": -1.0,
        "Death":        -1.0,
        "Gap":           0.0,   # or np.nan if you drop them earlier
    }

    last_idx = (
        df.groupby(["subject_id", "stay_id", "hfnc_episode"]).tail(1).index
    )
    df.loc[last_idx, "reward"] += df.loc[last_idx, "episode_outcome"].map(outcome_map)

    # ------------------------------------------------------------------
    # 9. Replace NaNs in spo2_next (episode ends) so r_A / r_B don’t propagate
    # ------------------------------------------------------------------
    df.fillna({"r_A": 0, "r_B": 0, "spo2_next": df["spo2"]}, inplace=True)

    # ------------------------------------------------------------------
    # 10. Final rescaling to [-1, 1]
    # ------------------------------------------------------------------
    max_abs = df["reward"].abs().max()
    if max_abs > 0:
        df["reward"] /= max_abs

    return df

def add_episode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a categorical `episode_outcome` for every HFNC episode.
    Priority (best → worst):
        Success  <  NIV  <  InvasiveVent  <  Death
    """

    # ───────────────────────────────────────── 1️⃣ map immediate transition
    vent_map = {
        "None":               "Success",
        "SupplementalOxygen": "Success",
        "NonInvasiveVent":    "NIV",
        "Tracheostomy":       "InvasiveVent",
        "InvasiveVent":       "InvasiveVent",
        "MechanicalVent":     "InvasiveVent",
        "Death":              "Death",        # if encoded directly
        "HFNC":               "Gap",          # HFNC→HFNC = data gap
    }
    df["episode_outcome"] = df["ventilation_to"].map(vent_map).fillna("Gap")

    # ───────────────────────────────────────── 2️⃣ last episode override
    # your stay-level outcome lives in the original column `outcome_label`
    if "outcome_label" in df.columns:
        is_last_ep = df["hfnc_episode"] == (
            df.groupby(["subject_id", "stay_id"])["hfnc_episode"].transform("max")
        )
        died_mask = df["outcome_label"].str.lower().eq("death")
        df.loc[is_last_ep & died_mask, "episode_outcome"] = "Death"

    return df

if __name__ == "__main__":
    main(debug=True)
