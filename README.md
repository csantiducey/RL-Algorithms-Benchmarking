# Bench that RLearner using Stable-Baselines3 🤖

Coding Template for a RL Pipeline that tests multiple learning algorithms (off-policy & on-policy) in a defined gymansium or custom environment. 

⭐ Ideal for beginners in the area of Reinforcement Learning who are looking to learn by developing and testing different available learning algorithms, without having the need to possess an extensive mathematical or expertise background.

## 📋 Table of Contents

-  [About the Project](#-about-the-project)
    -[Folder Structure](#folder-structure)
-  [Prerequisites](#prerequisites)
    -  [Installation](#installation)
-  [Documentation](#documentation)
-  [Licensing & Contact](#licensing--contact)


## About the Project 

The following software module provides the neccesary pipeline built over open-source RL frameworks to tune, test and compare the behavior of multiple RL learning algorithms and their learning performance over several custom environments.

The pipeline is built on top of the following integration:

**Gymnasium** _ _(environment library)_ _: Provides the simulated environments for agent training. Supports default small computationally-demanding environments, as well user-defined ones. 
> Full documentation can be found in (#documentation) section.

**Stable Baselines3** _ _(algorithm library)_ _: Provides the abstraction for the most used RL algorithms including but not limited to A2C, DQN, PPO, DDPG, SAC, TD3, etc. 

### 📖 Folder Structure
```text
├── README.md
├── eval_logs_best_a2c
│   ├── best_model.zip
│   └── evaluations.npz
├── eval_logs_best_dqn
│   ├── best_model.zip
│   └── evaluations.npz
├── fn_testing.py
├── models
│   └── LunarLander_best.zip
├── parameters.py
├── requirements.txt
├── testing.py
└── utils.py
```

+ **eval_logs_best_a2c** -> Stores the best trained agent, its callback performance as well as the learning curve using the on-policy A2C algorithm.

+ **eval_logs_best_dqn** -> Stores the best trained agent, its callback performance as well as the learning curve using the off-policy DQN algorithm.

+ **fn_testing.py** -> Unittest module for testing proper function behavior.

+ **parameters.py** -> Hash Tables containing the optimized parameters for agent training.

+ **requirements.txt** -> Needed depencies and packages for proper project run.

+ **testing.py** -> Main file containing the project pipeline (Instantiation, Training, Evaluation, Rendering).

+ **utils.py** -> Contains additional helper functions for visualization.

## Prerequisites

It is recommended to have some general basic knowledge of what is Reinforcement Learning, the architectures behind it, and a high-overview of the mathematical foundations that enable each learning algorithm. Object-Oriented Programming is a must. 

### ✅ Installation

Run the following commands for proper environment setup and package installation. 

In zsh terminal (macOS/Linux):

1. Defined desired workspace
```zsh
cd path/to/your/folder
```
2. Generate folder to duplicate GitHub repository
```zsh
mkdir RL_Benchmarking
```
3. Clone remote repository locally
```zsh
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
```
4. Create virtual environment
```zsh
python3 -m venv <venv_name>                               
```
5. Activate virtual environment
```zsh
source <venv_name>/bin/activate                            
```
6. Install required dependencies
```zsh
pip install -r requirements.txt                           
```

## 🧠 Documentation

Here are the full documentations for both frameworks used, as well as the papers describing the state-of-the-art learning algorithms.

    - Stable Baselines3: https://stable-baselines3.readthedocs.io/en/master/
    - Gymnasium: https://gymnasium.farama.org/index.html

    - A2C: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).

    - DDPG: [Deterministic Policy Gradient](https://proceedings.mlr.press/v32/silver14.pdf)

    - DQN: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

    - PPO: [Proximal Policy Optimization Algorithms](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)


## ✉️ Licensing & Contact

Santiago Ducey - [LinkedIn Profile](https://www.linkedin.com/in/csantiducey/) / santiagoduceychavez@gmail.com

I've you've reached this far, never let your dreams go away.



