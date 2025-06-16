import pandas as pd
import matplotlib.pyplot as plt
import os

def load_training_data(log_dir):
    """Load training data from d3rlpy CSV logs"""
    
    # Load the three CSV files
    td_loss_df = pd.read_csv(os.path.join(log_dir, 'td_loss.csv'), 
                            header=None, names=['epoch', 'step', 'value'])
    
    conservative_loss_df = pd.read_csv(os.path.join(log_dir, 'conservative_loss.csv'), 
                                      header=None, names=['epoch', 'step', 'value'])
    
    validation_td_error_df = pd.read_csv(os.path.join(log_dir, 'validation_td_error.csv'), 
                                        header=None, names=['epoch', 'step', 'value'])
    
    return td_loss_df, conservative_loss_df, validation_td_error_df

def plot_combined_training_curves(log_dir, save_path=None):
    """Create combined plot from d3rlpy log files"""
    
    # Load data
    td_loss_df, conservative_loss_df, validation_td_error_df = load_training_data(log_dir)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all three curves
    ax.plot(td_loss_df['step'], td_loss_df['value'], 
            label='Training TD Loss', color='blue', linewidth=2)
    ax.plot(conservative_loss_df['step'], conservative_loss_df['value'], 
            label='Training CQL Loss (Conservative)', color='red', linewidth=2)
    ax.plot(validation_td_error_df['step'], validation_td_error_df['value'], 
            label='Validation TD Error', color='green', linewidth=2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss / Error', fontsize=12)
    ax.set_title('CQL Training Dynamics: Loss Components & Validation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    return fig, ax

def plot_dual_axis_curves(log_dir, save_path=None):
    """Plot with dual y-axes for better scale handling"""
    
    # Load data
    td_loss_df, conservative_loss_df, validation_td_error_df = load_training_data(log_dir)
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Left axis: TD losses (smaller scale)
    color1 = 'tab:blue'
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('TD Loss/Error', color=color1, fontsize=12)
    
    line1 = ax1.plot(td_loss_df['step'], td_loss_df['value'], 
                     color='blue', label='Training TD Loss', linewidth=2)
    line2 = ax1.plot(validation_td_error_df['step'], validation_td_error_df['value'], 
                     color='green', label='Validation TD Error', linewidth=2)
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right axis: CQL loss (larger scale)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('CQL Loss (Conservative)', color=color2, fontsize=12)
    
    line3 = ax2.plot(conservative_loss_df['step'], conservative_loss_df['value'], 
                     color='red', label='Training CQL Loss', linewidth=2)
    
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)
    
    plt.title('CQL Training Dynamics: All Loss Components', fontsize=14, fontweight='bold')
    
    # Clean up spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    return fig, (ax1, ax2)

def plot_subplots_version(log_dir, save_path=None):
    """Plot in separate subplots for clearest view"""
    
    # Load data
    td_loss_df, conservative_loss_df, validation_td_error_df = load_training_data(log_dir)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Training TD Loss
    ax1.plot(td_loss_df['step'], td_loss_df['value'], 
             color='blue', linewidth=2)
    ax1.set_ylabel('Training TD Loss', fontsize=11)
    ax1.set_title('CQL Training Dynamics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Training CQL Loss  
    ax2.plot(conservative_loss_df['step'], conservative_loss_df['value'], 
             color='red', linewidth=2)
    ax2.set_ylabel('Training CQL Loss\n(Conservative)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Validation TD Error
    ax3.plot(validation_td_error_df['step'], validation_td_error_df['value'], 
             color='green', linewidth=2)
    ax3.set_ylabel('Validation TD Error', fontsize=11)
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()
    return fig, (ax1, ax2, ax3)

# Usage:
log_dir = "/Users/floppie/Documents/Msc Scriptie/HFNC codebase/first_pipeline/d3rlpy_logs/runs/cql_20250615170121"

# Option 1: Combined plot (might be hard to read due to scale differences)
plot_combined_training_curves(log_dir, save_path="combined_training_curves.png")

# Option 2: Dual axis plot (recommended for different scales)
plot_dual_axis_curves(log_dir, save_path="dual_axis_training_curves.png")

# Option 3: Subplot version (clearest view)
plot_subplots_version(log_dir, save_path="subplot_training_curves.png")