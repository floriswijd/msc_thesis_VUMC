from d3rlpy.dataset import Episode
from d3rlpy.algos import QLearningAlgoBase   # Fixed import for d3rlpy 2.8.1
from d3rlpy.metrics import TDErrorEvaluator
from scope_rl.policy.head import SoftmaxHead
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import BaseHead, SoftmaxHead
from scope_rl.ope import CreateOPEInput, OffPolicyEvaluation
from scope_rl.ope.discrete import TrajectoryWiseImportanceSampling, DoublyRobust, SelfNormalizedDR
from scope_rl.policy import EpsilonGreedyHead     
import yaml

# --- Step 1: Define the Conversion Helper Function (The key to making it easy) ---
def convert_d3rlpy_to_scope_rl(d3rlpy_dataset: d3rlpy.dataset.MDPDataset) -> OfflineDataset:
    """A helper function to seamlessly convert a d3rlpy dataset to a scope-rl dataset."""
    terminals = d3rlpy_dataset.terminals.astype(bool)
    scope_rl_dataset = OfflineDataset(
        state=d3rlpy_dataset.observations,
        action=d3rlpy_dataset.actions,
        reward=d3rlpy_dataset.rewards,
        done=terminals,
    )
    print("Successfully converted d3rlpy dataset to scope-rl format.")
    return scope_rl_dataset
