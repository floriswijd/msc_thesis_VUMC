import pandas as pd
import yaml
from pathlib import Path
############ old
TARGET_SPO2 = 94           # centre of desired range
SETPOINT_K  = 2          # ±k → reward falls to 0
FLOW_COST   = 0.002        # per L/min
FIO2_COST   = 0.005        # per % (use 0.5 % if you prefer frac)
###########

# --- Clinically-Motivated Constants for Reward Function (v7 - Combined Terminal Outcome) ---
# SpO2 Targets
TARGET_SPO2_MIN = 92.0
TARGET_SPO2_MAX = 96.0
SPO2_CRITICAL_LOW = 88.0
SPO2_ACCEPTABLE_MIN_POST_ACTION = 92.0

# Cost Factors
FIO2_COST_FACTOR = -0.005
FLOW_COST_FACTOR = -0.002
HIGH_FIO2_THRESHOLD = 60.0
HIGH_FIO2_PENALTY = -0.2
HIGH_FLOW_THRESHOLD = 50.0
HIGH_FLOW_PENALTY = -0.1

# Step Reward Magnitudes
REWARD_IN_OPTIMAL_SPO2_RANGE = 1.0
PENALTY_SPO2_BELOW_CRITICAL = -2.0
PENALTY_MILD_HYPEROXIA_FACTOR = -0.2
REWARD_MAINTAINING_STABILITY = 0.5
PENALTY_DETERIORATION_IF_STABLE = -0.5

# Terminal Outcome Values (for the transition at the end of HFNC episode)
TERMINAL_REWARD_TRANSITION_SUCCESS = +1.0  # e.g., to Room Air or Supplemental O2
TERMINAL_PENALTY_TRANSITION_NIV = -0.5
TERMINAL_PENALTY_TRANSITION_IMV = -1.0
TERMINAL_REWARD_TRANSITION_GAP = 0.0    # For HFNC -> HFNC or unmapped transitions

# Additional Penalty if the *overall hospital stay* associated with the *last* HFNC episode resulted in Death
TERMINAL_PENALTY_STAY_DEATH_ADDITIONAL = -1.5 # This is a strong additional penalty

# Weights for step reward components
WEIGHT_OXYGENATION = 0.50
WEIGHT_CLINICAL_STABILITY = 0.30
WEIGHT_RESOURCE_UTILIZATION = 0.20

RAW_CSV  = Path("HFNC codebase/first_pipeline/data/feature_engineered_data.csv")
OUT_PARQ = Path("HFNC codebase/first_pipeline/data/hfnc_episodes.parquet")

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
    cat_cols  = ["rox_class", "humidification"]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
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
    """
    Calculates rewards (Version 7).
    Terminal reward combines immediate transition outcome and an additional penalty
    if the hospital stay (for the last HFNC episode) resulted in death.
    """
    df = df.sort_values(
        ["subject_id", "stay_id", "hfnc_episode", "hour_ts"]
    ).reset_index(drop=True)

    # Required columns for step rewards + 'episode_transition_outcome' for terminal
    required_cols = ["spo2", "fio2", "o2_flow", "rox_class", "sf_ratio", "episode_transition_outcome"]
    if "outcome_label" not in df.columns: # Needed for death penalty
        print("Warning: 'outcome_label' column not found. Cannot apply additional death penalty.")
        
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column for reward calculation: {col}")

    group_cols = ["subject_id", "stay_id", "hfnc_episode"]
    df["spo2_next"] = df.groupby(group_cols)["spo2"].shift(-1)
    df["spo2_next"].fillna(df["spo2"], inplace=True)

    df["mask_high_risk"] = (
        (df["rox_class"] == "high") |
        (df["spo2"] < 90) | 
        (df["sf_ratio"] < 235)
    )

    # --- Component 1: Oxygenation Reward ---
    df["r_oxygenation"] = 0.0
    df.loc[(df["spo2_next"] >= TARGET_SPO2_MIN) & (df["spo2_next"] <= TARGET_SPO2_MAX), "r_oxygenation"] = REWARD_IN_OPTIMAL_SPO2_RANGE
    df.loc[(df["spo2_next"] >= SPO2_CRITICAL_LOW) & (df["spo2_next"] < TARGET_SPO2_MIN), "r_oxygenation"] = \
        (df["spo2_next"] - TARGET_SPO2_MIN) / (TARGET_SPO2_MIN - SPO2_CRITICAL_LOW) 
    df.loc[df["spo2_next"] < SPO2_CRITICAL_LOW, "r_oxygenation"] = PENALTY_SPO2_BELOW_CRITICAL
    df.loc[df["spo2_next"] > TARGET_SPO2_MAX, "r_oxygenation"] = \
        PENALTY_MILD_HYPEROXIA_FACTOR * (df["spo2_next"] - TARGET_SPO2_MAX) / (100.0 - TARGET_SPO2_MAX)
    df["r_oxygenation"] = df["r_oxygenation"].clip(PENALTY_SPO2_BELOW_CRITICAL, REWARD_IN_OPTIMAL_SPO2_RANGE)

    # --- Component 2: Clinical Stability ---
    df["r_clinical_stability"] = 0.0
    is_currently_stable = (df["mask_high_risk"] == False) & (df["rox_class"] == "low")
    maintains_stability_post_action = (df["spo2_next"] >= SPO2_ACCEPTABLE_MIN_POST_ACTION)
    df.loc[is_currently_stable & maintains_stability_post_action, "r_clinical_stability"] = REWARD_MAINTAINING_STABILITY
    df.loc[(df["mask_high_risk"] == False) & (df["spo2_next"] < SPO2_ACCEPTABLE_MIN_POST_ACTION), "r_clinical_stability"] = PENALTY_DETERIORATION_IF_STABLE
    
    # --- Component 3: Resource Utilization ---
    base_cost = (FIO2_COST_FACTOR * df["fio2"]) + (FLOW_COST_FACTOR * df["o2_flow"])
    high_fio2_penalty = df["fio2"].apply(lambda x: HIGH_FIO2_PENALTY if x > HIGH_FIO2_THRESHOLD else 0.0)
    high_flow_penalty = df["o2_flow"].apply(lambda x: HIGH_FLOW_PENALTY if x > HIGH_FLOW_THRESHOLD else 0.0)
    df["r_resource_utilization"] = base_cost + high_fio2_penalty + high_flow_penalty
    df["r_resource_utilization"] = df["r_resource_utilization"].clip(-1.0, 0.0)

    # --- Sum weighted step reward components ---
    df["reward"] = (
        WEIGHT_OXYGENATION * df["r_oxygenation"] +
        WEIGHT_CLINICAL_STABILITY * df["r_clinical_stability"] +
        WEIGHT_RESOURCE_UTILIZATION * df["r_resource_utilization"]
    )

    # --- Add Terminal Rewards (Combined Logic) ---
    outcome_map_transition = { # Based on 'episode_transition_outcome' from ventilation_to
        "Success": TERMINAL_REWARD_TRANSITION_SUCCESS,
        "NIV": TERMINAL_PENALTY_TRANSITION_NIV,
        "InvasiveVent": TERMINAL_PENALTY_TRANSITION_IMV,
        "Gap": TERMINAL_REWARD_TRANSITION_GAP, 
    }
    
    last_idx = df.groupby(group_cols).tail(1).index
    
    # 1. Apply base terminal reward based on the HFNC episode's immediate transition outcome
    df.loc[last_idx, "reward"] += df.loc[last_idx, "episode_transition_outcome"].map(outcome_map_transition).fillna(TERMINAL_REWARD_TRANSITION_GAP)
    
    # 2. Apply additional penalty if the hospital stay outcome was death AND this was the last HFNC episode of that stay
    if "outcome_label" in df.columns:
        # Mark if this HFNC episode is the last one for its stay_id
        df['is_last_hfnc_ep_in_stay'] = df.groupby(['subject_id', 'stay_id'])['hfnc_episode'].transform('max') == df['hfnc_episode']
        
        # Identify indices that are:
        #   a) The last step of an episode (already identified by last_idx)
        #   b) Part of the last HFNC episode in the stay
        #   c) Where the hospital outcome_label (for the stay) is "Death"
        condition_for_death_penalty = (
            df.index.isin(last_idx) &
            df['is_last_hfnc_ep_in_stay'] &
            (df['outcome_label'].astype(str).str.lower() == 'death')
        )
        df.loc[condition_for_death_penalty, "reward"] += TERMINAL_PENALTY_STAY_DEATH_ADDITIONAL
        
        df.drop(columns=['is_last_hfnc_ep_in_stay'], inplace=True, errors='ignore')
    else:
        print("DEBUG: 'outcome_label' not found, skipping additional death penalty for terminal rewards.")


    # --- Final Rescaling of the total reward ---
    max_abs_reward = df["reward"].abs().max()
    if max_abs_reward > 0:
        df["reward"] = df["reward"] / max_abs_reward
    df["reward"] = df["reward"].clip(-1.0, 1.0)
    
    return df

def add_episode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a categorical `episode_transition_outcome` for every HFNC episode (Version 3).
    This outcome is based *only* on the `ventilation_to` column, representing the
    immediate transition from HFNC. It does not consider the overall hospital `outcome_label` here.
    """
    if "ventilation_to" not in df.columns:
        print("Warning: 'ventilation_to' column missing in add_episode_outcome_v3. Will fill 'episode_transition_outcome' with 'Gap'.")
        df["episode_transition_outcome"] = "Gap"
        return df

    # Map `ventilation_to` to immediate episode transition outcomes.
    # "Death" is not expected in `ventilation_to`.
    # NaN/None in `ventilation_to` is mapped to "Success" (assumed weaned to room air).
    # "HFNC" in `ventilation_to` (HFNC -> HFNC) is mapped to "Gap".
    vent_map_transition = {
        "None": "Success",                 # Weaned to room air/no support
        "SupplementalOxygen": "Success",   # Weaned to lower level oxygen
        "NonInvasiveVent": "NIV",          # Escalated to NIV
        "Tracheostomy": "InvasiveVent",    # Escalated to Trach (form of IMV)
        "InvasiveVent": "InvasiveVent",    # Escalated to IMV
        "MechanicalVent": "InvasiveVent",  # Escalated to IMV (synonym)
        "HFNC": "Gap",                     # HFNC to HFNC (ongoing or data artifact)
    }
    
    # Fill NaN in 'ventilation_to' with "None" before mapping, assuming it means successful weaning.
    df['ventilation_to_filled'] = df['ventilation_to'].fillna("None")
    df["episode_transition_outcome"] = df['ventilation_to_filled'].map(vent_map_transition)
    
    # If any `ventilation_to_filled` values were not in vent_map_transition, they will be NaN. Fill these with "Gap".
    df["episode_transition_outcome"].fillna("Gap", inplace=True)
    df.drop(columns=['ventilation_to_filled'], inplace=True, errors='ignore')
    
    return df



def add_rewards_old(df: pd.DataFrame) -> pd.DataFrame:
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

def add_episode_outcome_old(df: pd.DataFrame) -> pd.DataFrame:
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
