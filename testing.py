'''RL Algorithm Testing'''

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C, DQN, TD3, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env, VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from utils import plot_learning_curve
from parameters import A2C_optimized as a2c
from parameters import DQN_optimized as dqn

# Environments Documentations
# (Discrete) -> "LunarLander-v3" : https://gymnasium.farama.org/environments/box2d/lunar_lander/
# (Continuous) -> "CarRacing-v3" : https://gymnasium.farama.org/environments/box2d/car_racing/

# Optimized Environment Parameters (retrieved from Docs)
env_id = "LunarLander-v3"
n_training_envs = 8
n_eval_envs = 5

# Optimized Algorithm Parameters found in parameters.py


def main_process(seed: int = 0) -> None:
    # Generate vectorized training environments (On-Policy and Off-Policy must not share)
    train_env_a2c = make_vec_env(env_id, n_envs=n_training_envs, seed=seed, vec_env_cls=SubprocVecEnv)

    train_env_dqn = make_vec_env(env_id, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

    # Instantiate distinct Parallelized Evaluation Environments
    eval_env_a2c = make_vec_env(env_id, n_envs=n_eval_envs, seed=seed, vec_env_cls=DummyVecEnv)
    eval_env_dqn = make_vec_env(env_id, n_envs=n_eval_envs, seed=seed, vec_env_cls=DummyVecEnv)


    # Create callback that evaluates agent
    a2c_callback = EvalCallback(eval_env_a2c, best_model_save_path=eval_log_dir_a2c,
                                log_path=eval_log_dir_a2c, eval_freq=max(a2c["n_timesteps"] // n_training_envs, 1), 
                                deterministic=True, render=False, verbose=0
                                )

    dqn_callback = EvalCallback(eval_env_dqn, best_model_save_path=eval_log_dir_dqn,
                                log_path=eval_log_dir_dqn, eval_freq=200000,
                                deterministic=True, render=False, verbose=0
                                )

    # Instantiate and generate Models with Optimized Parameters
    model_a2c = A2C(policy  =   a2c["policy"], 
                    env     =   train_env_a2c, 
                    learning_rate = a2c["learning_rate"],         
                    n_steps =   a2c["n_steps"],                                
                    gamma   =   a2c["gamma"],                                
                    ent_coef = a2c["ent_coef"], 
                    vf_coef =   a2c["vf_coef"],               
                    device  =   "cpu",                               # A2C optimized for CPU processing
                    normalize_advantage = True,
                    use_rms_prop    =   False,
                    policy_kwargs   =   a2c["policy_kwargs"],
                    seed=0,                        
                    verbose=0
                )

    model_dqn = DQN(policy  = dqn["policy"],
                    env     = train_env_dqn,
                    buffer_size     =   dqn["buffer_size"],           
                    batch_size      =   dqn["batch_size"],                    
                    gamma   =   dqn["gamma"],
                    learning_starts =   dqn["learning_starts"],
                    train_freq      =   dqn["train_freq"],
                    gradient_steps  =   dqn["gradient_steps"],
                    target_update_interval  =   dqn["target_update_interval"],
                    exploration_fraction    =   dqn["exploration_fraction"],
                    exploration_final_eps   =   dqn["exploration_final_eps"],
                    policy_kwargs   =   dqn["policy_kwargs"],
                    device="auto",
                    seed=0,
                    verbose=0
                )

    # Train Models
    print("\n------ STARTING TO TRAIN MODELS ------\n")

    print("Training A2C...")
    model_a2c.learn(total_timesteps=a2c["n_timesteps"], callback=a2c_callback, progress_bar=True) # 5,000,000 steps / 8 environments = 625,000 steps
    print("A2C Model trained ✅")
    
    print("\n Training DQN Model...")
    model_dqn.learn(total_timesteps=dqn["n_timesteps"], callback=dqn_callback, progress_bar=True)
    print("DQN Model trained ✅")

    print("\n📈 Evaluating Policy Performance...")
    # Compare and save Best Model
    a2c_mean, a2c_std = evaluate_policy(model=model_a2c, 
                                        env=eval_env_a2c,
                                        n_eval_episodes=10,
                                        deterministic=True
                                        )
    dqn_mean, dqn_std = evaluate_policy(model=model_dqn,
                                        env=eval_env_dqn,
                                        n_eval_episodes=10,
                                        deterministic=True
                                        )
    
    print("\n---POLICY PERFORMANCE EVALUATION (10 episodes)---")

    print(f"\nA2C Model -> Mean Reward: {a2c_mean:.2f} +/- {a2c_std:.2f}")
    print(f"\nDQN Model -> Mean Reward: {dqn_mean:.2f} +/- {dqn_std:.2f}")

    # Learning Curve Rendering
    plot_learning_curve(eval_log_dir_a2c)
    plot_learning_curve(eval_log_dir_dqn)

    # Render Best Model
    best_path = "models/LunarLander_best"

    if a2c_mean > dqn_mean :
        best_model = model_a2c
    else : 
        best_model = model_dqn

    best_model.save(best_path)
        
    print('\n🏆 Rendering Best Model Behavior...')

    vec_env = best_model.get_env()
    obs = vec_env.reset()

    while True : 
        action, _states = best_model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
        if dones.any():
            break

    print("\n💯 Simulation finished, clossing all windows...")

    vec_env.close()

    return None
    

if __name__ == '__main__' :

    # Create log dirs to save performance metrics
    eval_log_dir_a2c = "./eval_logs_best_a2c/"
    eval_log_dir_dqn = "./eval_logs_best_dqn/"
    os.makedirs(eval_log_dir_a2c, exist_ok=True)
    os.makedirs(eval_log_dir_dqn, exist_ok=True)

    main_process()



