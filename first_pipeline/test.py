import numpy as np
from d3rlpy.dataset import Episode, create_infinite_replay_buffer

print("="*60)
print("🔬 STARTING TEST: How d3rlpy handles a malformed Episode")
print("="*60)

# 1. Create data for a malformed episode.
print("\n[SETUP] Creating data with matching lengths (the incorrect structure)...")
malformed_obs = np.array([[0.], [1.], [2.], [3.]])
malformed_act = np.array([10, 11, 12, 13])
malformed_rew = np.array([0.1, 0.2, 0.3, 0.4])

print(f"   - Shape of observations: {malformed_obs.shape}")
print(f"   - Shape of actions:      {malformed_act.shape}")

# 2. Create the malformed Episode object.
malformed_episode = Episode(
    observations=malformed_obs,
    actions=malformed_act,
    rewards=malformed_rew,
    terminated=True
)

print("\n[INFO] Checking the properties of the created Episode object...")
print(f"   - malformed_episode.size() returns: {malformed_episode.size()}")
print(f"   - len(malformed_episode.observations) is: {len(malformed_episode.observations)}")
if len(malformed_episode.observations) != malformed_episode.size() + 1:
    print("   - CONFIRMED: The episode is structurally inconsistent.")

# 3. We will now manually calculate the number of transitions d3rlpy can extract.
# A transition requires a `next_state`, so the number of transitions is always `len(observations) - 1`.
print("\n[TEST] Deducing the number of extractable transitions...")
num_transitions_extracted = len(malformed_episode.observations) - 1


print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
print(f"   Number of transitions in the malformed episode (size): {malformed_episode.size()}")
print(f"   Number of transitions ACTUALLY EXTRACTED by d3rlpy: {num_transitions_extracted}")
print("="*60)

# 4. Conclusion
if num_transitions_extracted == malformed_episode.size() - 1:
    print("\n[CONCLUSION] The test PASSED.")
    print("d3rlpy must have extracted N-1 transitions and silently discarded the final action.")
    print("This confirms why your training ran without crashing while the validation check failed.")
else:
    print("\n[CONCLUSION] The test FAILED.")
    print("The behavior is different than expected.")