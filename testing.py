'''RL Algorithm Testing'''

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Set environment / render_mode="human" pops a visual window
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Instantiate, train and save agent model
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(5e4), progress_bar=True)
model.save("a2c_lunar")
del model

# Load Trained Agent
model = A2C.load("a2c_lunar", env=env, print_system_info=False)

vec_env = model.get_env()

# Evaluate Agent Learning
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)

print('---AGENT BEHAVIOR---')
print(f' -> MEAN reward: {mean_reward:.2f}')
print(f' -> STD  reward: {std_reward:.2f}')

# Observe trained behavior for 2000 steps
obs = vec_env.reset()
for iteration in range(2000) :
    # Iterate until timestep limit reached or out-of bound conditions reached.
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)   
    # print("Done DataType:", done)
    vec_env.render("human")
    
env.close()









