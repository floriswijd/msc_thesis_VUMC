import numpy as np, pandas as pd, itertools, json
from typing import List, Tuple

# ---------- load the processed parquet ------------------------
df = pd.read_parquet("hfnc_episodes.parquet")        # <- your path
flow, fio2 = df["o2_flow"], df["fio2"]

def make_edges(vals: pd.Series, n_bins: int, snap=5) -> List[int]:
    """Quantile edges snapped to nearest `snap`."""
    qs = np.linspace(0, 1, n_bins + 1)
    raw = np.unique(np.quantile(vals, qs).round())   # raw quantiles
    snapped = np.unique((np.round(raw / snap) * snap).astype(int))
    snapped[0], snapped[-1] = vals.min(), vals.max() + 1
    return snapped.tolist()

def score_bins(f_edges: List[int], o_edges: List[int],
               min_cnt=500, max_imbalance=10) -> Tuple[float, pd.DataFrame]:
    # digitise
    f_idx = np.digitize(flow, bins=f_edges, right=False) - 1
    o_idx = np.digitize(fio2, bins=o_edges, right=False) - 1
    counts = pd.crosstab(f_idx, o_idx)
    cnt_vals = counts.values.ravel()
    support_violation = np.sum(cnt_vals < min_cnt)
    imbalance = cnt_vals.max() / cnt_vals.min()
    n_actions = counts.size
    cost = 100 * support_violation + 10 * (imbalance - 1) + (n_actions - 15) ** 2
    return cost, counts

def grid_search(flow_bins_opts=(3,4),
                fio2_bins_opts=(4,5,6),
                verbose=True):
    best = {"cost": 1e9}
    for nb_f, nb_o in itertools.product(flow_bins_opts, fio2_bins_opts):
        f_edges = make_edges(flow, nb_f)
        o_edges = make_edges(fio2, nb_o)
        cost, table = score_bins(f_edges, o_edges)
        if verbose:
            print(f"{nb_f}×{nb_o} bins → cost {cost:.1f}")
        if cost < best["cost"]:
            best.update(cost=cost, f_edges=f_edges, o_edges=o_edges,
                        counts=table)
    return best


best = grid_search()
print("\n=== Best candidate ===")
print("Flow edges :", best['f_edges'])
print("FiO₂ edges :", best['o_edges'])
print("Action count:", best['counts'].size)
print(best['counts'])         # Jupyter nice table


import seaborn as sns, matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(best["counts"], annot=True, fmt="d", cmap="viridis")
plt.xlabel("FiO₂ bin"); plt.ylabel("Flow bin")
plt.title("Candidate balanced bins")
plt.tight_layout()
plt.savefig("candidate_bins_heatmap.png", dpi=300)
plt.show()