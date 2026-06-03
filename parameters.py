# OPTIMIZED LEARNING PARAMETERS FOR RL ALGORITHMS

A2C_optimized = {

            "n_timesteps" : int(5e6),
            "policy"      : "MlpPolicy",

            "n_steps"     : 32,             # Rollout Buffer 32 x 8 = 256 steps / update
            "learning_rate" : 0.00083,
            "gamma"       : 0.999,
            "normalize_advantage" : True,

            "ent_coef"    : 0.01,
            "vf_coef"     : 0.4,

            "use_rms_prop" : False,

            # --Network Architecture--

             "policy_kwargs" : dict(
            net_arch = dict(
            pi = [256, 256],                # Separate actor network
            vf = [256, 256],                # Separate critic network
        )
    )
}

DQN_optimized = {

            "n_timesteps" : int(1e6),
            "policy"      : "MlpPolicy",

            "gamma"       : 0.99,
            "buffer_size" : int(1e5),
            "batch_size"  : 128,
            "learning_starts" : int(1e4),

            "train_freq"  : 4,             # Update every 4 env steps
            "gradient_steps" : -1,
            "target_update_interval" : 250,
            
            "exploration_fraction" : 0.12,
            "exploration_final_eps" : 0.10,

            # ── Network architecture ───────────────────────────────────────
            "policy_kwargs" : dict(
        net_arch = [256, 256]               # Two hidden layers of 256 units
    )
}
