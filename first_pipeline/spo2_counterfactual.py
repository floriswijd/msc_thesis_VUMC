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

# ---------------------------------------------------------------------------
# Configuration dataclass – keeps the knobs for quick experimentation
# ---------------------------------------------------------------------------
@dataclass
class DynConfig:
    """Hyper‑parameters for the GradientBoostingRegressor dynamics model."""

    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 5
    loss: str = "quantile"   # use "quantile" for 50‑th percentile (=median)
    alpha: float = 0.5        # target quantile when loss == "quantile"
    test_size: float = 0.2
    random_state: int = 0


# ---------------------------------------------------------------------------
# Public helper 1/2 – train the forward SpO₂ model
# ---------------------------------------------------------------------------

def train_spo2_dynamics(
    train_eps: Sequence["Episode"],
    n_actions: int,
    spo2_idx: int,
    cfg: DynConfig | None = None,
    obs_scaler: "obs_scaler" | None = None,  # type: ignore – d3rlpy obs scaler
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
        s         = ep.observations[:-1]                # (T‑1, d)
        a         = ep.actions[:-1].reshape(-1).astype(int)   # (T-1,)
        next_spo2 = ep.observations[1:, spo2_idx]       # (T‑1,)

        sa = np.hstack([s, eye[a]])                    # concat one‑hot action
        X.append(sa)
        y.append(next_spo2)

    X = np.vstack(X)
   
    y = np.concatenate(y)
    row_idx = np.arange(len(X))   
    print("First 5 rows before split:", row_idx[:5])
    # --- train/validation split ---------------------------------------------
    Xtr, Xva, ytr, yva, idx_tr, idx_va = train_test_split(
        X, y,row_idx,   test_size=cfg.test_size, random_state=cfg.random_state,shuffle=False,  #don't shuffle the data before training
    )

    model = GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        loss=cfg.loss,
        alpha=cfg.alpha,
        random_state=cfg.random_state,
  
    )
    model.fit(Xtr, ytr)

    print("First 5 train-row indices:", idx_tr[:5])
    print("First 5 val-row  indices:", idx_va[:5])


    mean  = obs_scaler.mean[spo2_idx]
    scale = obs_scaler.std[spo2_idx]
    yva_raw     = yva * scale + mean
    pred_raw    = model.predict(Xva) * scale + mean
    mae_raw     = mean_absolute_error(yva_raw, pred_raw)
    print(f"[SpO2-Dyn] Hold-out MAE (raw units): {mae_raw:.2f} %")

    # quick sanity‑check
    mae = mean_absolute_error(yva, model.predict(Xva))
    r2       = r2_score(yva,model.predict(Xva))
    print(f"[SpO2-Dyn] R² (expl. var.)   : {r2:.3f}")
    print(f"[SpO2‑Dyn] Hold‑out MAE: {mae:.3f} %")

    return model


# ---------------------------------------------------------------------------
# Public helper 2/2 – roll‑out an episode under CQL actions
# ---------------------------------------------------------------------------

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
                                  n_actions: int):
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
    cql_act[0] = -1                 # dummy action at t=0

    for t in range(1, T):
        a_id         = int(cql_model.predict(state.reshape(1, -1))[0])
        cql_act[t]   = a_id
        sa           = np.hstack([state, eye[a_id]])
        next_spo2    = float(dyn_model.predict(sa.reshape(1, -1))[0])

        next_state             = ep.observations[t].copy()
        next_state[spo2_idx]   = next_spo2   # overwrite *only* SpO₂
        spo2_cf[t]             = next_spo2
        state                  = next_state

    return spo2_cf, cql_act


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
    ax_act.set_ylabel("Action ID"); ax_act.set_yticks(range(n_actions))
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


__all__ = [
    "DynConfig",
    "train_spo2_dynamics",
    "rollout_cql_episode",
    "plot_spo2_trajectories",
    "rollout_cql_episode_spo2_only",
    "plot_actions_and_spo2",
]