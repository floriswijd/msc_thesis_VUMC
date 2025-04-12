# main.py
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from env import HFNCEnvironment
from q_learning_agent import QLearningAgent

def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def main():
    data_path = 'data/preprocessed_data_reduced_3.csv'  # update with your CSV file path
    env = HFNCEnvironment(data_path)

    # Define a simple action space:
    # 0: No change, 1: Increase HFNC flow, 2: Decrease HFNC flow,
    # 3: Increase FiO₂, 4: Decrease FiO₂
    action_space = [0, 1, 2, 3, 4]
    agent = QLearningAgent(action_space=action_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    
    num_epochs = 10  # Number of passes (episodes) for training
    state_visitation_counts = defaultdict(int)  # For tracking state frequencies
    episode_rewards = []  # For tracking total reward per episode
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        state = env.reset()
        episode_total_reward = 0.0
        done = False
        
        while not done:
            # Log the visitation count for each composite state.
            discrete_state = agent.discretize_state(state)
            state_visitation_counts[discrete_state] += 1
            
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_total_reward += reward
        
        print(f"Epoch {epoch+1} completed with total reward {episode_total_reward:.4f}")
        episode_rewards.append(episode_total_reward)
    
    # Save the Q-table for future analysis.
    agent.save_q_table("q_table.json")
    
    # Get a timestamp string for file naming.
    ts = get_timestamp()

    # -------------------------
    # Plot State Visitation Frequency
    # -------------------------
    states = list(state_visitation_counts.keys())
    frequencies = [state_visitation_counts[s] for s in states]
    
    # Sort states by frequency (highest first)
    sorted_idx = sorted(range(len(frequencies)), key=lambda i: frequencies[i], reverse=True)
    sorted_states = [states[i] for i in sorted_idx]
    sorted_frequencies = [frequencies[i] for i in sorted_idx]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_states)), sorted_frequencies)
    plt.xlabel("Composite States")
    plt.ylabel("Visitation Frequency")
    plt.title("State Visitation Frequency")
    plt.xticks(range(len(sorted_states)), sorted_states, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"state_visitation_{ts}.png")
    plt.show()

    # -------------------------
    # Plot Q-Value Distribution for Visited States
    # -------------------------
    q_values_all = []
    for state in states:
        q_vals = agent.Q_table[state]
        q_values_all.extend(q_vals.tolist())
    
    plt.figure(figsize=(8, 6))
    plt.hist(q_values_all, bins=20, edgecolor='black')
    plt.xlabel("Q-Value")
    plt.ylabel("Frequency")
    plt.title("Q-Value Distribution (All Actions, All Visited States)")
    plt.tight_layout()
    plt.savefig(f"q_value_distribution_{ts}.png")
    plt.show()
    
    # -------------------------
    # Plot Learning Curve: Total Reward per Episode
    # -------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), episode_rewards, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve: Total Reward per Epoch")
    plt.tight_layout()
    plt.savefig(f"learning_curve_{ts}.png")
    plt.show()

if __name__ == "__main__":
    main()