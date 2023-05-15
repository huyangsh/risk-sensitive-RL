import numpy as np
from stable_baselines3 import PPO, SAC, DQN, TD3
import gym
import os
import argparse

import random


from env import RMDP, CartPole, Pendulum, build_toy_10_env, build_toy_100_env, build_toy_1000_env
from data import Dataset, TorchDataset


env_name = "CartPole"
alg_name = "random"
SIGMA = 0.01
alg_path = "./data/expert_alg/Pendulum_SAC.zip"
data_path = f"./data/{env_name}/{env_name}_{SIGMA}_{alg_name}.pkl"
SIZE = int(1e6)
DRL_algs = {"PPO": PPO, "SAC": SAC, "DQN": DQN, "TD3": TD3}
DRL_envs = {"CartPole": CartPole, "Pendulum": Pendulum}

parser = argparse.ArgumentParser()
# OpenAI gym environment name (need to be consistent with the dataset name)
parser.add_argument("--env", default="CartPole-v1")
# e-mix (prob. to mix random actions)
parser.add_argument("--eps", default=1.0, type=float)
parser.add_argument("--buffer_size", default=1e6, type=float)
parser.add_argument("--verbose", default="False", type=str)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--gendata_pol", default="ppo", type=str)
args = parser.parse_args()


if env_name == "RMDP":
    p_perturb = 0.15
    beta  = 0.01
    gamma = 0.95
    env = build_toy_1000_env(p_perturb, beta, gamma)

    if alg_name == "random":
        policy = lambda x: random.choice(env.actions)
    elif alg_name == "expert":
        best_response = [0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 0, 0]
        policy = lambda x: best_response[x]
    else:
        raise NotImplementedError
    
    T_eval = SIZE
    _, trajectory = env.eval(policy, T_eval=T_eval, verbose=True)

    dataset = Dataset()
    dataset.store(trajectory)
    dataset.save("./data/Toy/toy_1000_random.npy")
elif env_name == "RMDP_torch":
    p_perturb = 0.15
    beta  = 0.01
    gamma = 0.95
    env = build_toy_100_env(p_perturb, beta, gamma)

    if alg_name == "random":
        policy = lambda x: random.choice(env.actions)
    elif alg_name == "expert":
        best_response = [0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 0, 0]
        policy = lambda x: best_response[x]
    else:
        raise NotImplementedError

    dataset = TorchDataset(device="cpu")
    dataset.start(1, 1, SIZE)

    T_eval = 200
    while dataset.size < SIZE:
        _, trajectory = env.eval(policy, T_eval=T_eval, verbose=True)
        for s,a,r,s_,f in trajectory: dataset.add(s, a, r, s_, f)
        print(f"add {len(trajectory)} data points, total = {dataset.size}")
        for i in range(3):
            data = random.choice(trajectory)
            print(f"sample data #{i}: {data}.")
    
    dataset.finish()
    dataset.save("./data/Toy/toy_torch_random.pkl")
elif env_name in ["CartPole", "Pendulum"]:
    env = DRL_envs[env_name](sigma=SIGMA)
    print(f"Successfully build {env_name}.")
    
    if alg_name == "random":
        policy = lambda x: random.choice(env.actions)
    elif alg_name in DRL_algs:
        epsilon = 0.3
        agent = DRL_algs[alg_name].load(alg_path, device=args.device)
        policy = lambda x: random.choice(env.actions) if np.random.binomial(n=1, p=epsilon) else agent.predict(x)[0]
    else:
        raise NotImplementedError
    
    dataset = TorchDataset(device="cpu")
    dataset.start(env.dim_state, env.dim_action, SIZE)

    T_eval = 200
    while dataset.size < SIZE:
        _, trajectory = env.eval(policy, T_eval=T_eval, verbose=True)
        for s,a,r,s_,f in trajectory: dataset.add(s, a, r, s_, f)
        print(f"add {len(trajectory)} data points, total = {dataset.size}")
        for i in range(3):
            data = random.choice(trajectory)
            print(f"sample data #{i}: {data}.")
    
    dataset.finish()
    dataset.save(data_path)
else:
    raise NotImplementedError