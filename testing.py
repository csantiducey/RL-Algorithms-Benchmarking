'''RL Algorithm Testing'''

import gymnasium as gym
from stable_baselines3 import A2C

# Set environment / render_mode="human" pops a visual window
env = gym.make("LunarLander-v3", render_mode="human")
env.reset()

# Train specified model
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test trainin behavior
for iteration in range(10) :

    obs, info = env.reset()
    
    # Reset ending flags for every iteration
    terminated, truncated = False, False

    while not terminated and not truncated: 
        # Iterate until timestep limit reached or out-of bound conditions reached.
        env.render()
        action = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    
env.close()









