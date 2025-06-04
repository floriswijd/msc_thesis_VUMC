import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Load data
df = pd.read_parquet("hfnc_episodes.parquet")
counts = df["action"].value_counts().sort_index()

# Load configuration from YAML file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract parameter mappings from config
flow_edges = config['flow_edges']
fio2_edges = config['fio2_edges']
n_actions = config['n_actions']

def create_action_mapping():
    """Create mapping from action ID to human-readable HFNC parameters"""
    mapping = {}
    action = 0
    
    for flow_idx in range(len(flow_edges) - 1):
        for fio2_idx in range(len(fio2_edges) - 1):
            flow_min, flow_max = flow_edges[flow_idx], flow_edges[flow_idx + 1] - 1
            fio2_min, fio2_max = fio2_edges[fio2_idx], fio2_edges[fio2_idx + 1] - 1
            
            # Handle edge case for maximum values
            if flow_max == 70:  # Last flow bin goes to 70
                flow_max = 70
            if fio2_max == 100:  # Last FiO2 bin goes to 100
                fio2_max = 100
                
            mapping[action] = {
                'flow_range': f"{flow_min}–{flow_max}",
                'fio2_range': f"{fio2_min}–{fio2_max}",
                'short_label': f"F:{flow_min}–{flow_max}\nO₂:{fio2_min}–{fio2_max}%",
                'full_label': f"{flow_min}–{flow_max} L/min, {fio2_min}–{fio2_max}% FiO₂"
            }
            action += 1
    
    return mapping

# Verify configuration consistency
print(f"Config verification:")
print(f"- Flow edges: {flow_edges}")
print(f"- FiO₂ edges: {fio2_edges}")
print(f"- Expected actions: {n_actions}")
print(f"- Calculated actions: {(len(flow_edges)-1) * (len(fio2_edges)-1)}")
assert n_actions == (len(flow_edges)-1) * (len(fio2_edges)-1), "Config mismatch!"

# Create the mapping
action_mapping = create_action_mapping()

print("\nAction ID to HFNC Parameter Mapping (from config.yaml):")
print("=" * 60)
for action_id in sorted(action_mapping.keys()):
    mapping = action_mapping[action_id]
    print(f"Action {action_id:2d}: {mapping['full_label']}")

print(f"\nAction distribution summary:")
print(counts)
print(f"\nMost common action: Action {counts.idxmax()} ({action_mapping[counts.idxmax()]['full_label']})")
print(f"Most common action share: {counts.max()/counts.sum():.1%}")

# Create enhanced visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Standard action histogram with readable labels
actions = sorted(counts.index)
action_labels = [action_mapping[a]['short_label'] for a in actions]
action_counts = [counts[a] for a in actions]

bars1 = ax1.bar(range(len(actions)), action_counts, alpha=0.7, 
                color=plt.cm.viridis(np.linspace(0, 1, len(actions))))
ax1.set_xlabel("HFNC Parameter Combinations")
ax1.set_ylabel("Number of Transitions")
ax1.set_title("Action Distribution: HFNC Flow Rate and FiO₂ Combinations\n(Configuration from config.yaml)")
ax1.set_xticks(range(len(actions)))
ax1.set_xticklabels(action_labels, rotation=45, ha='right', fontsize=9)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars1, action_counts)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(action_counts)*0.01,
             f'{count}', ha='center', va='bottom', fontsize=8)

# Plot 2: Heatmap showing flow vs FiO2 combinations
flow_bins = len(flow_edges) - 1
fio2_bins = len(fio2_edges) - 1
heatmap_data = np.zeros((flow_bins, fio2_bins))

for action_id, count in counts.items():
    flow_idx = action_id // fio2_bins
    fio2_idx = action_id % fio2_bins
    heatmap_data[flow_idx, fio2_idx] = count

# Create flow and FiO2 labels from config
flow_labels = [f"{flow_edges[i]}–{flow_edges[i+1]-1 if i < len(flow_edges)-2 else flow_edges[i+1]}" 
               for i in range(len(flow_edges)-1)]
fio2_labels = [f"{fio2_edges[i]}–{fio2_edges[i+1]-1 if i < len(fio2_edges)-2 else fio2_edges[i+1]}" 
               for i in range(len(fio2_edges)-1)]

im = ax2.imshow(heatmap_data, cmap='viridis', aspect='auto')
ax2.set_xlabel("FiO₂ Range (%)")
ax2.set_ylabel("Flow Rate Range (L/min)")
ax2.set_title("Action Frequency Heatmap: Flow Rate vs FiO₂\n(From config.yaml)")
ax2.set_xticks(range(len(fio2_labels)))
ax2.set_xticklabels(fio2_labels)
ax2.set_yticks(range(len(flow_labels)))
ax2.set_yticklabels(flow_labels)

# Add text annotations to heatmap
for i in range(flow_bins):
    for j in range(fio2_bins):
        action_id = i * fio2_bins + j
        if action_id in counts:
            ax2.text(j, i, f'{int(heatmap_data[i, j])}', 
                    ha="center", va="center", color="white", fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Number of Transitions')

plt.tight_layout()
plt.savefig("enhanced_action_histogram_from_config.png", dpi=300, bbox_inches='tight')
plt.show()

# Generate summary table for thesis appendix
print(f"\n{'='*80}")
print("THESIS APPENDIX: Complete Action Mapping Table (from config.yaml)")
print(f"{'='*80}")
print(f"{'Action ID':<10} {'Flow Rate (L/min)':<20} {'FiO₂ (%)':<15} {'Frequency':<12} {'Percentage':<10}")
print(f"{'-'*80}")

total_transitions = counts.sum()
for action_id in sorted(action_mapping.keys()):
    mapping = action_mapping[action_id]
    freq = counts.get(action_id, 0)
    pct = (freq / total_transitions * 100) if total_transitions > 0 else 0
    print(f"{action_id:<10} {mapping['flow_range']:<20} {mapping['fio2_range']:<15} "
          f"{freq:<12} {pct:<10.1f}%")

print(f"{'-'*80}")
print(f"{'TOTAL':<10} {'':<35} {total_transitions:<12} {'100.0%':<10}")
print(f"\nConfiguration source: config.yaml")
print(f"Flow edges: {flow_edges}")
print(f"FiO₂ edges: {fio2_edges}")