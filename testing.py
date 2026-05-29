'''RL Algorithm Testing'''

import os
import gymnasium as gym
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from utils import plot_learning_curve

# Environments Documentations
# (Discrete) -> "LunarLander-v3" : https://gymnasium.farama.org/environments/box2d/lunar_lander/
# (Continuous) -> "CarRacing-v3" : https://gymnasium.farama.org/environments/box2d/car_racing/

# Optimized Environment Parameters (retrieved from Docs)
env_id = "LunarLander-v3"
n_training_envs = 8
n_eval_envs = 5

# Optimized Model Parameters
n_timesteps = int(4e5)
learning_rate = 0.00083
gamma = 0.995
ent_coef = 0.0005
steps = 32
buffer_size = int(2e5)

# Create log dir to save evaluation results
eval_log_dir_a2c = "./eval_logs_best_a2c/"
eval_log_dir_dqn = "./eval_logs_best_dqn/"
os.makedirs(eval_log_dir_a2c, exist_ok=True)
os.makedirs(eval_log_dir_dqn, exist_ok=True)

def main_process() -> None:
    # Generate vectorized training environments (On-Policy and Off-Policy must not share)
    train_env_a2c = make_vec_env(env_id, n_envs=n_training_envs, seed=0, vec_env_cls=SubprocVecEnv)

    train_env_dqn = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=DummyVecEnv)

    # Shared Parallelized Evaluation Environments
    eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0, vec_env_cls=DummyVecEnv)


    # Create callback that evaluates agent
    a2c_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir_a2c,
                                log_path=eval_log_dir_a2c, eval_freq=max(int(1e5) // n_training_envs, 1), 
                                deterministic=True, render=False
                                )

    dqn_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir_dqn,
                                log_path=eval_log_dir_dqn, eval_freq=max(25000 // n_training_envs, 1),
                                deterministic=True, render=False
                                )

    # Instantiate and generate Models with Optimized Parameters
    model_a2c = A2C("MlpPolicy", 
                    env=train_env_a2c, 
                    learning_rate=learning_rate,         # O.00083
                    n_steps=steps,                       # 32 -> Increment batch_size for smoother updating
                    gamma=gamma,                         # 0.995 -> Highly Prioritize Future Reward
                    ent_coef=ent_coef,                   # 0.0001
                    device="cpu",                        # A2C optimized for CPU processing
                    seed=0,                        
                    verbose=0
                    )

    model_dqn = DQN("MlpPolicy",
                    env=train_env_dqn,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,            # 200,000 -> Bigger the better, limited only by computer RAM
                    batch_size=256,                     # Experiences samples before gradient update
                    gamma=gamma,
                    train_freq=64,
                    device="auto",
                    seed=0,
                    verbose=0
                    )

    # Train Models
    print("------ STARTING TO TRAIN MODELS ------\n")

    print("Training A2C...")
    model_a2c.learn(total_timesteps=n_timesteps, callback=a2c_callback, progress_bar=True) # 400,000 steps / 8 environments = 50,000 steps x env
    print("A2C Model trained ✅")
    
    print("\n Training DQN Model...")
    model_dqn.learn(total_timesteps=int(1e5), callback=dqn_callback, progress_bar=True)
    print("DQN Model trained ✅")

    print("\n📈 Evaluating Policy Performance...")
    # Compare and save Best Model
    a2c_mean, a2c_std = evaluate_policy(model=model_a2c, 
                                        env=eval_env,
                                        n_eval_episodes=10,
                                        deterministic=True
                                        )
    dqn_mean, dqn_std = evaluate_policy(model=model_dqn,
                                        env=eval_env,
                                        n_eval_episodes=10,
                                        deterministic=True
                                        )
    
    print("\n---POLICY PERFORMANCE EVALUATION (10 episodes)---")

    print(f"\nA2C Model -> Mean Reward: {a2c_mean:.2f} +/- {a2c_std:.2f}")
    print(f"\nDQN Model -> Mean Reward: {dqn_mean:.2f} +/- {dqn_std:.2f}")

    # Learning Curve Rendering
    plot_learning_curve(f"{eval_log_dir_a2c}evaluations.npz")
    plot_learning_curve(f"{eval_log_dir_dqn}evaluations.npz")

    # Render Best Model
    path = "models/LunarLander_best"

    if a2c_mean > dqn_mean :
        best_model = model_a2c
    else : 
        best_model = model_dqn

    best_model.save(path)
        
    vec_env = best_model.get_env()

    print('\n🏆 Rendering Best Model Behavior...')

    vec_env = best_model.get_env()
    obs = vec_env.reset()

    while True : 
        action, _states = best_model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones.any():
            break

    vec_env.close()

    return None
    

if __name__ == '__main__' :
    main_process()











