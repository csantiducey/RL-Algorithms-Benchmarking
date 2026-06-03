# ADDITIONAL UTIL FUNCTIONS
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(npz_path: str) -> None:
    # Load the compressed numpy archive
    data = np.load(npz_path + "evaluations.npz")
    
    # Extract evaluation timesteps and raw episodic rewards
    timesteps = data["timesteps"]
    results = data["results"]  # Shape: (num_evaluations, n_eval_episodes)
    
    # Calculate statistics across the evaluation episodes (axis 1)
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    # Plot configuration
    plt.figure(figsize=(10, 6))
    
    # Plot the mean evaluation reward line
    plt.plot(timesteps, mean_rewards, label="Mean Eval Reward", color="blue", linewidth=2)
    
    # Fill the region representing ±1 Standard Deviation for variance visualization
    plt.fill_between(
        timesteps, 
        mean_rewards - std_rewards, 
        mean_rewards + std_rewards, 
        color="blue", 
        alpha=0.15, 
        label="Variance (±1 SD)"
    )
    
    # Format labels and titles for LunarLander
    plt.title("LunarLander-v3 Evaluation Learning Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Training Timesteps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    


    # Save Figure
    plt.savefig(npz_path)

    # Display Figure
    plt.tight_layout()
    plt.show()

    return None
