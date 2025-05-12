#!/usr/bin/env python3
# -----------------------------------------------------------
# train_cql.py  --  Offline Conservative Q-Learning voor HFNC
# -----------------------------------------------------------
#
# • Laadt data/processed/hfnc_episodes.parquet
# • Maakt MDPDataset met train/val/test splits
# • Instantieert DiscreteCQL (d3rlpy) met hyperparams uit YAML
# • Logt FQE en WIS tijdens training
# • Slaat model + scaler + metrics op
#
# CLI:
#   python train_cql.py --alpha 1.0 --epochs 200 --gpu 0
# -----------------------------------------------------------

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import StandardScaler
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics import wis_evaluator, wdr_evaluator

# ---------- CLI ARGUMENTEN ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data",  default="data/processed/hfnc_episodes.parquet")
parser.add_argument("--cfg",   default="config.yaml")
parser.add_argument("--alpha", type=float, default=1.0, help="CQL conservatisme-gewicht")
parser.add_argument("--epochs", type=int,  default=200)
parser.add_argument("--batch",  type=int,  default=256)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--gamma",  type=float, default=0.99)
parser.add_argument("--gpu",    type=int,   default=0, help="-1 = CPU")
parser.add_argument("--logdir", default="runs/cql")
args = parser.parse_args()

# ---------- PADEN ----------
Path(args.logdir).mkdir(parents=True, exist_ok=True)
model_path  = Path(args.logdir) / "cql_hfnc.pt"
metric_path = Path(args.logdir) / "metrics.yaml"

# ---------- DATA INLADEN ----------
print("⏳  Loading Parquet …")
df  = pd.read_parquet(args.data)
cfg = yaml.safe_load(open(args.cfg))

state_cols = [c for c in df.columns
              if c not in ("action","reward","done",
                           "subject_id","stay_id","hfnc_episode",
                           "outcome_label","hour_ts")]

states  = df[state_cols].values.astype("float32")
actions = df["action"].values.astype("int64")
rewards = df["reward"].values.astype("float32")
dones   = df["done"].values.astype("bool")

dataset = MDPDataset(states, actions, rewards, dones)

# ---------- SPLIT train / val / test ----------
train_eps, temp_eps = train_test_split(dataset.episodes,
                                       test_size=0.3,
                                       random_state=42)
val_eps,   test_eps = train_test_split(temp_eps,
                                       test_size=0.5,
                                       random_state=42)

print(f"📊 Episodes  train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")

# ---------- SCALER ----------
scaler = StandardScaler()   # z-score voor alle features

# ---------- CQL INSTANTIE ----------
cql = DiscreteCQL(
    action_size=int(cfg["n_actions"]),
    alpha=args.alpha,
    batch_size=args.batch,
    lr=args.lr,
    gamma=args.gamma,
    encoder_factory="dense",
    hidden_units=[256, 256],
    n_epochs=args.epochs,
    scaler=scaler,
    use_gpu=(args.gpu >= 0),
    gpu_id=args.gpu
)

# ---------- TRAIN ----------
print("🚀  Start training DiscreteCQL …")
cql.fit(
    train_eps,
    eval_episodes=val_eps,
    logdir=args.logdir,
    scorers={
        "FQE": cql.evaluate_behavior_policy,
        "WIS": wis_evaluator,
    },
)

# ---------- OFFLINE TEST EVALUATIE ----------
fqe_test = cql.evaluate_behavior_policy(test_eps)
wis_test = wis_evaluator(cql, test_eps)
wdr_test = wdr_evaluator(cql, test_eps)

metrics = dict(FQE_test=float(fqe_test),
               WIS_test=float(wis_test),
               WDR_test=float(wdr_test),
               alpha=args.alpha,
               epochs=args.epochs)

yaml.safe_dump(metrics, open(metric_path, "w"))
print("✅  Test-scores:", metrics)

# ---------- MODEL OPSLAAN ----------
cql.save_model(model_path)
print(f"💾  Model opgeslagen → {model_path}")
