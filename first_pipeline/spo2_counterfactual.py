# -*- coding: utf-8 -*-
"""spo2_counterfactual.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utility helpers to **learn a forward SpO₂ dynamics model** from the
offline dataset and to roll‑out counter‑factual SpO₂ trajectories when a
policy (e.g. your trained CQL agent) acts in place of the clinician.

Drop this file into the project root and import the two public helpers
below from your evaluation script or notebook:

>>> from spo2_counterfactual import train_spo2_dynamics, rollout_cql_episode

The minimal workflow is:
    1. dyn_model = train_spo2_dynamics(train_eps, n_actions, spo2_idx)
    2. cf_spo2   = rollout_cql_episode(ep, cql, dyn_model, spo2_idx, n_actions)

© 2025, OpenAI Demo – free to use for academic purposes.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
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
        a         = ep.actions[:-1].astype(int)         # (T‑1,)
        next_spo2 = ep.observations[1:, spo2_idx]       # (T‑1,)

        sa = np.hstack([s, eye[a]])                    # concat one‑hot action
        X.append(sa)
        y.append(next_spo2)

    X = np.vstack(X)
    y = np.concatenate(y)

    # --- train/validation split ---------------------------------------------
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
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

    # quick sanity‑check
    mae = mean_absolute_error(yva, model.predict(Xva))
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


__all__ = [
    "DynConfig",
    "train_spo2_dynamics",
    "rollout_cql_episode",
    "plot_