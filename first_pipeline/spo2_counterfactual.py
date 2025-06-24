# -*- coding: utf-8 -*-
"""spo2_counterfactual.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility helpers to **learn a forward SpO₂ dynamics model** from the
offline dataset and to roll‑out counter‑factual SpO₂ trajectories when a
policy (e.g. your trained CQL agent) acts in place of the clinician.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from typing import Tuple
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuration dataclass – keeps the knobs for quick experimentation
# ---------------------------------------------------------------------------
@dataclass
class DynConfig:
    """Hyper‑parameters for the GradientBoostingRegressor dynamics model."""

    n_estimators: int = 600
    learning_rate: float = 0.1
    max_depth: int = 4
    loss: str = "quantile"   # use "quantile" for 50‑th percentile (=median)
    alpha: float = 0.5        # target quantile when loss == "quantile"
    test_size: float = 0.2
    random_state: int = 2


# ---------------------------------------------------------------------------
# Public helper 1/2 – train the forward SpO₂ model
# ---------------------------------------------------------------------------
# pull the cut-points you wrote in config.yaml   :contentReference[oaicite:2]{index=2}
FLOW_EDGES = np.array([0, 20, 40, 71],  dtype=int)     # [0–20) [20–40) [40–70]
FIO2_EDGES = np.array([21, 40, 60, 80, 101], dtype=int)# [21–40) [40–60) [60–80) [80–100]
N_FLOW_BINS = len(FLOW_EDGES) - 1          # 3
N_FIO2_BINS = len(FIO2_EDGES) - 1         
N_FIO2  = len(FIO2_EDGES) - 1              # 4
N_FLOW  = len(FLOW_EDGES) - 1              # 3
N_ACT   = N_FLOW * N_FIO2                  # 12



def action_id_to_label(a_id: int) -> str:
    """Return a human-readable (flow, FiO₂) bin for a discrete action id."""
    flow_idx = a_id // N_FIO2_BINS
    fio2_idx = a_id %  N_FIO2_BINS
    f_low, f_hi   = FLOW_EDGES[flow_idx],   FLOW_EDGES[flow_idx + 1]
    o2_low, o2_hi = FIO2_EDGES[fio2_idx],   FIO2_EDGES[fio2_idx + 1]
    return f"{f_low}-{f_hi} L | {o2_low}-{o2_hi} %"

def id_to_midpoints(a: np.ndarray | int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map discrete action id(s) → mid-point flow- and FiO₂-values.

    Parameters
    ----------
    a : scalar id or 1-D int array of shape (T,)

    Returns
    -------
    flow_mid  : np.ndarray, same shape as a   (L min⁻¹)
    fio2_mid  : np.ndarray, same shape as a   (%)
    """
    a = np.asarray(a, dtype=int).ravel()
    flow_idx   = a // N_FIO2
    fio2_idx   = a %  N_FIO2

    flow_mid   = 0.5 * (FLOW_EDGES[flow_idx]   + FLOW_EDGES[flow_idx+1])
    fio2_mid   = 0.5 * (FIO2_EDGES[fio2_idx]   + FIO2_EDGES[fio2_idx+1])
    return flow_mid.reshape(a.shape), fio2_mid.reshape(a.shape)

def train_spo2_dynamics(
    train_eps: Sequence["Episode"],
    n_actions: int,
    spo2_idx: int,
    *,
    cfg: DynConfig | None = None,
    obs_scaler: "obs_scaler" | None = None,  # type: ignore – d3rlpy obs scaler
    base_feat_idx: np.ndarray | None = None, # index of the first base feature in the observation vector (default 0)
):
    """Fit a one‑step predictor `SpO2_{t+1} = f(s_t, a_t)`.

    Parameters
    ----------
    train_eps : iterable of d3rlpy.dataset.Episode
        Episodes from the **training split** (do *not* include test episodes).
    n_actions : int
        Cardinality of the discrete action space (e.g. 12).
    spo2_idx  : int
        Column index of SpO₂ in the observation/state vectors.
    cfg       : DynConfig | None
        Optionally override default hyper‑parameters.

    Returns
    -------
    model : sklearn.ensemble.GradientBoostingRegressor
        Trained regressor.  Call `.predict(X)` for scalar output; if you
        set `loss='quantile'`, the predictions correspond to the chosen
        quantile (`alpha`).
    """
    if cfg is None:
        cfg = DynConfig()

    

    # --- build feature matrix ------------------------------------------------
    X, y = [], []
    eye = np.eye(n_actions)


    for ep in train_eps:
        T = len(ep)
        if T < 4:
            print("too short")
            continue                     # guard, though you filtered earlier

        s_t = ep.observations[2:T-1][:, base_feat_idx]
        s_tm1 = ep.observations[1:T-2][:, base_feat_idx]
        s_tm2 = ep.observations[0:T-3][:, base_feat_idx]
        a_t   = ep.actions[2:T-1].reshape(-1).astype(int)
        y_tp1 = ep.observations[3:T, spo2_idx]

        lag1_spo2 = s_tm1[:, spo2_idx]
        lag2_spo2 = s_tm2[:, spo2_idx]
        # lag1_rox  = s_tm1[:, rox_idx]

        # sa = np.hstack([ s_t,
        #                 eye[a_t],
        #                 lag1_spo2[:, None],
        #                 lag2_spo2[:, None],
        #                 lag1_rox[:,  None] ])
        
        # flow_mid, fio2_mid = id_to_midpoints(a_t)

        sa = np.hstack([
                s_t,
                eye[a_t],                  # 12 sparse cols
                # flow_mid[:, None],         # 1 dense col  ← NEW
                # fio2_mid[:, None],         # 1 dense col  ← NEW
                lag1_spo2[:, None],
                lag2_spo2[:, None]
                # lag1_rox[:, None]
        ])


        X.append(sa)
        y.append(y_tp1)

    X = np.vstack(X)          # (N, d+12+3)
    y = np.concatenate(y)     # (N,)
    print("X rows:", X.shape[0], "y rows:", y.shape[0])  # should match


    row_idx = np.arange(len(X))   
    print("First 5 rows before split:", row_idx[:5])
    # --- train/validation split ---------------------------------------------
    Xtr, Xva, ytr, yva, idx_tr, idx_va = train_test_split(
        X, y,row_idx,   test_size=cfg.test_size, random_state=cfg.random_state,shuffle=False,  #don't shuffle the data before training
    )

    # model = GradientBoostingRegressor(
    #     n_estimators=cfg.n_estimators,
    #     learning_rate=cfg.learning_rate,
    #     max_depth=cfg.max_depth,
    #     loss=cfg.loss,
    #     alpha=cfg.alpha,
    #     random_state=cfg.random_state,
  
    # )
    model = XGBRegressor(
        n_estimators       = cfg.n_estimators,
        learning_rate      = cfg.learning_rate,
        max_depth          = cfg.max_depth,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        objective          = "reg:squarederror",
        random_state       = cfg.random_state,
        n_jobs             = -1,
        
)

    model.fit(Xtr, ytr)

    print("First 5 train-row indices:", idx_tr[:5])
    print("First 5 val-row  indices:", idx_va[:5])


    # mean  = obs_scaler.mean[spo2_idx]
    # scale = obs_scaler.std[spo2_idx]
    # yva_raw     = yva * scale + mean
    # pred_raw    = model.predict(Xva) * scale + mean
    # mae_raw     = mean_absolute_error(yva_raw, pred_raw)
    # print(f"[SpO2-Dyn] Hold-out MAE (raw units): {mae_raw:.2f} %")

    # quick sanity‑check
    mae = mean_absolute_error(yva, model.predict(Xva))
    r2       = r2_score(yva,model.predict(Xva))
    print(f"[SpO2-Dyn] R² (expl. var.)   : {r2:.3f}")
    print(f"[SpO2‑Dyn] Hold‑out MAE: {mae:.3f} %")

    return model


# ---------------------------------------------------------------------------
# Public helper 2/2 – roll‑out an episode under CQL actions
# ---------------------------------------------------------------------------

def assemble_features(obs, act, spo2_idx,  eye):
    if act.ndim > 1:                      # ← NEW
        act = act.reshape(-1)             # ← NEW  make it 1-D

    lag2_spo2 = obs[0:-3, spo2_idx]
    lag1_spo2 = obs[1:-2, spo2_idx]
    # lag1_rox  = obs[1:-2, rox_idx]
    s_t       = obs[2:-1]
    a_t       = act[2:-1]

    return np.hstack([s_t,
                      eye[a_t],
                      lag1_spo2[:, None],
                      lag2_spo2[:, None]]
                      )

def rollout_cql_episode(
    ep: "Episode",
    cql_model,  # type: ignore – d3rlpy algorithm with .predict()
    dyn_model: GradientBoostingRegressor,
    spo2_idx: int,
    n_actions: int,
):
    """Simulate SpO₂ trajectory if the **CQL policy** had driven the episode.
 
    The function keeps *all* other observed features fixed (i.e. it is a
    *one‑variable counter‑factual*).  For full‑state simulation you would
    train a model for every physiological variable and stitch them into a
    joint next‑state predictor – out of scope for this quick demo.

    Returns
    -------
    np.ndarray, shape (T,)
        Counter‑factual SpO₂ values for each timestep t=0…T‑1.
    """
    T = len(ep)
    state = ep.observations[0].copy()
    eye = np.eye(n_actions)

    spo2_cf = np.empty(T)
    spo2_cf[0] = state[spo2_idx]

    for t in range(1, T):
        # 1) choose the action the agent *would* take in this state
        a = int(cql_model.predict(state.reshape(1, -1))[0])   


        # 2) predict SpO2_{t} under that (state, action)
        sa = np.hstack([state, eye[a]])
        next_spo2 = float(dyn_model.predict(sa.reshape(1, -1))[0])

        # 3) create next‑state (copy all observed features, overwrite SpO₂)
        next_state = ep.observations[t].copy()
        next_state[spo2_idx] = next_spo2

        spo2_cf[t] = next_spo2
        state = next_state  # roll forward

    return spo2_cf


# ---------------------------------------------------------------------------
# Convenience plotting function (optional)
# ---------------------------------------------------------------------------

def plot_spo2_trajectories(
    time_points: Sequence[int],
    spo2_obs: Sequence[float],
    spo2_pred_clin: Sequence[float],
    spo2_pred_cql: Sequence[float],
    title: str | None = None,
    save_path: str | None = None,
):
    """Overlay observed & model‑predicted SpO₂ trajectories.

    Parameters
    ----------
    time_points : iterable of int
        X‑axis ticks (e.g. np.arange(T)).
    spo2_obs : observed SpO₂ (clinician path).
    spo2_pred_clin : model‑predicted SpO₂ under *clinician* actions.
    spo2_pred_cql  : model‑predicted SpO₂ under *CQL* actions.
    title, save_path : optional aesthetics.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3))
    plt.plot(time_points, spo2_obs, label="Observed SpO₂ (clin)", c="grey")
    plt.plot(time_points, spo2_pred_clin, label="Pred SpO₂ (clin)", c="red")
    plt.plot(time_points, spo2_pred_cql, label="Pred SpO₂ (CQL)", c="purple", ls="--")
    plt.hlines([92, 96], xmin=time_points[0], xmax=time_points[-1],
               colors="green", linestyles=":", alpha=0.3)
    plt.ylim(85, 100)
    plt.ylabel("SpO₂ (%)")
    plt.xlabel("Time step")
    if title:
        plt.title(title)
    plt.legend(frameon=False)
    plt.grid(alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def rollout_cql_episode_spo2_only(ep, cql_model, dyn_model,
                                  spo2_idx: int,
                                  n_actions: int,
                                  rox_idx: int,):

    """
    Returns
    -------
    spo2_cf : (T,) counter-factual SpO₂ under CQL roll-out
    cql_act : (T,) greedy CQL action id at each step (t=0 is dummy -1)
    """
    eye        = np.eye(n_actions)
    state      = ep.observations[0].copy()
    T          = len(ep)
    spo2_cf    = np.empty(T)
    cql_act    = np.empty(T, dtype=int)

    spo2_cf[0] = state[spo2_idx]
    cql_act[0] = cql_act[0] = int(cql_model.predict(state.reshape(1, -1))[0])              # dummy action at t=0

    lag2 = state[spo2_idx]
    lag1 = state[spo2_idx]          # both equal at t=0

    for t in range(1, T):
        a_id         = int(cql_model.predict(state.reshape(1, -1))[0])
        cql_act[t]   = a_id
        # sa           = np.hstack([state, eye[a_id]])
        sa = np.hstack([ state, eye[a_id],
                     lag1,   # SpO₂_{t-1}
                     lag2,   # SpO₂_{t-2}
                     state[rox_idx] ])  # ROX_{t-1} (from current state)
        next_spo2    = float(dyn_model.predict(sa.reshape(1, -1))[0])

        next_state             = ep.observations[t].copy()
        next_state[spo2_idx]   = next_spo2   # overwrite *only* SpO₂
        spo2_cf[t]             = next_spo2

        lag2, lag1 = lag1, next_spo2            # shift buffer
        state                  = next_state

    return spo2_cf, cql_act

# NEW 2 axises def plot_actions_and_spo2( 
#         t,
#         flow_c, fio2_c, flow_q, fio2_q,
#         spo2_obs, spo2_pred_clin, spo2_pred_cql,
#         title: str | None = None,
#         save: str | None = None):
#     """
#     Panel-1 : Flow-rate (primary y) + FiO₂ (twin y), clinician vs CQL
#     Panel-2 : Observed & predicted SpO₂
#     """
#     import matplotlib.pyplot as plt

#     fig, (ax_par, ax_o2) = plt.subplots(
#         2, 1, figsize=(12, 5),
#         sharex=True, height_ratios=[1.4, 1.8])

#     # -- panel 1: parameters --------------------------------------------
#     ax_par.step(t, flow_c, c="red",  where="mid", label="Flow Clin")
#     ax_par.step(t, flow_q, c="blue", where="mid", label="Flow CQL")
#     ax_par.set_ylabel("Flow (L/min)")
#     ax_par.grid(alpha=.3)

#     ax_twin = ax_par.twinx()
#     ax_twin.step(t, fio2_c, c="red",  where="mid", ls="--", label="FiO₂ Clin")
#     ax_twin.step(t, fio2_q, c="blue", where="mid", ls="--", label="FiO₂ CQL")
#     ax_twin.set_ylabel("FiO₂ (%)")

#     # merged legend
#     h1, l1 = ax_par.get_legend_handles_labels()
#     h2, l2 = ax_twin.get_legend_handles_labels()
#     ax_par.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

#     # -- panel 2: SpO₂ ---------------------------------------------------
#     ax_o2.plot(t, spo2_obs,       c="grey",  label="Observed SpO₂")
#     ax_o2.plot(t, spo2_pred_clin, c="red",   label="Pred SpO₂ (clin)")
#     ax_o2.plot(t, spo2_pred_cql,  c="purple",ls="--", label="Pred SpO₂ (CQL)")
#     ax_o2.axhspan(92, 96, color="green", alpha=.12)
#     ax_o2.set_ylabel("SpO₂ (%)"); ax_o2.set_ylim(85, 100)
#     ax_o2.set_xlabel("Episode step")
#     ax_o2.legend(frameon=False); ax_o2.grid(alpha=.3)

#     if title: fig.suptitle(title, y=1.02)
#     plt.tight_layout()
#     if save:
#         plt.savefig(save, dpi=300)
#     else:
#         plt.show()

def predict_spo2_one_step(obs,            # (T, d) recorded states
                          actions,        # (T,) int ids to use
                          dyn_model,
                          spo2_idx: int,
                          n_actions: int, 
                          base_feat_idx: np.ndarray | None = None):
    """
    Return a length-T array of one-step SpO₂ predictions, conditioning
    on the *logged* state at every step (open-loop).
    """
    eye = np.eye(n_actions)
    # re-create the same feature matrix you trained on  (skip t<2 for lags)
    s_t = obs[2:-1][:, base_feat_idx]
    a_t      = actions[2:-1].reshape(-1)
    lag1_spo2 = obs[1:-2, spo2_idx]
    lag2_spo2 = obs[0:-3, spo2_idx]


    # X = np.hstack([s_t,
    #                eye[a_t],
    #                lag1_spo2[:, None],
    #                lag2_spo2[:, None],
    #                lag1_rox[:,  None]])
    # flow_mid, fio2_mid = id_to_midpoints(a_t)

    X = np.hstack([
                s_t,
                eye[a_t],                  # 12 sparse cols
                # flow_mid[:, None],         # 1 dense col  ← NEW
                # fio2_mid[:, None],         # 1 dense col  ← NEW
                lag1_spo2[:, None],
                lag2_spo2[:, None],
        ])


    pred = dyn_model.predict(X)  
    # print("DEBUG one-step  X.shape =", X.shape)   # <-- add this line once

    # prepend the first two logged values so length==T
    # return np.r_[obs[:2, spo2_idx], dyn_model.predict(X)]
    return np.r_[obs[:2, spo2_idx], pred, pred[-1]]



def plot_actions_and_spo2(ep,
                          clin_act, cql_act,
                          spo2_obs, spo2_pred_clin, spo2_pred_cql,
                          n_actions: int,
                          title: str | None = None,
                          save: str | None = None):
    """Two stacked panels: actions (top) and SpO₂ traces (bottom)."""
    t = np.arange(len(ep))
    fig, (ax_act, ax_o2) = plt.subplots(2, 1, figsize=(12, 5),
                                        sharex=True, height_ratios=[1, 1.6])

    # -- actions -----------------------------------------------------------
    ax_act.step(t, clin_act, where="mid", lw=2,          label="Clinician")
    ax_act.step(t, cql_act,  where="mid", lw=2, ls="--", label="CQL rollout")
    ax_act.set_ylabel("Action bin")
    ax_act.set_yticks(range(n_actions))

    # >>> ADD THESE TWO LINES <<<
    from spo2_counterfactual import action_id_to_label          # ⟵ NEW
    ax_act.set_yticklabels([action_id_to_label(i)               # ⟵ NEW
                            for i in range(n_actions)],
                           fontsize=8)

    ax_act.legend(frameon=False); ax_act.grid(alpha=.3)

    # -- SpO₂ --------------------------------------------------------------
    ax_o2.plot(t, spo2_obs,       c="grey",  label="Observed SpO₂")
    ax_o2.plot(t, spo2_pred_clin, c="red",   label="Pred SpO₂ (clin)")
    ax_o2.plot(t, spo2_pred_cql,  c="purple",ls="--", label="Pred SpO₂ (CQL)")
    ax_o2.axhspan(92, 96, color="green", alpha=.1)
    ax_o2.set_ylabel("SpO₂ (%)"); ax_o2.set_ylim(85, 100)
    ax_o2.set_xlabel("Episode step")
    ax_o2.legend(frameon=False); ax_o2.grid(alpha=.3)

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300)
    else:
        plt.show()

#OLD def plot_actions_and_spo2(ep,
#                           clin_act, cql_act,
#                           spo2_obs, spo2_pred_clin, spo2_pred_cql,
#                           n_actions: int,
#                           title: str | None = None,
#                           save: str | None = None):
#     """Two stacked panels: actions (top) and SpO₂ traces (bottom)."""
#     t = np.arange(len(ep))
#     fig, (ax_act, ax_o2) = plt.subplots(2, 1, figsize=(12, 5),
#                                         sharex=True, height_ratios=[1, 1.6])

#     # -- actions -----------------------------------------------------------
#     ax_act.step(t, clin_act, where="mid", lw=2,          label="Clinician")
#     ax_act.step(t, cql_act,  where="mid", lw=2, ls="--", label="CQL rollout")
#     ax_act.set_ylabel("Action ID"); ax_act.set_yticks(range(n_actions))
#     ax_act.legend(frameon=False); ax_act.grid(alpha=.3)

#     # -- SpO₂ --------------------------------------------------------------
#     ax_o2.plot(t, spo2_obs,       c="grey",  label="Observed SpO₂")
#     ax_o2.plot(t, spo2_pred_clin, c="red",   label="Pred SpO₂ (clin)")
#     ax_o2.plot(t, spo2_pred_cql,  c="purple",ls="--", label="Pred SpO₂ (CQL)")
#     ax_o2.axhspan(92, 96, color="green", alpha=.1)
#     ax_o2.set_ylabel("SpO₂ (%)"); ax_o2.set_ylim(85, 100)
#     ax_o2.set_xlabel("Episode step")
#     ax_o2.legend(frameon=False); ax_o2.grid(alpha=.3)

#     if title:
#         fig.suptitle(title, y=1.02)
#     plt.tight_layout()
#     if save:
#         plt.savefig(save, dpi=300)
#     else:
#         plt.show()    


__all__ = [
    "DynConfig",
    "train_spo2_dynamics",
    "rollout_cql_episode",
    "plot_spo2_trajectories",
    "rollout_cql_episode_spo2_only",
    "plot_actions_and_spo2",
]