import numpy as np
import random
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt

from env import RMDP, build_toy_env
from agent import RFZI_Tabular

THRES = 1e-5

seed = 0
random.seed(seed)
np.random.seed(seed)

# Build environment
p_perturb = 0.15
beta  = 0.01
gamma = 0.95
env = build_toy_env(p_perturb, beta, gamma, THRES)

# Load data.
dataset = np.load("./data/toy/toy_random_0.01.npy")

# Build agent.
agent = RFZI_Tabular(env)
Z_init = np.ones(shape=(env.num_states, env.num_actions), dtype=np.float64)
agent.reset(Z_init)

T = 10
for t in range(T): 
    _, info = agent.update(dataset)
    print(f"loss at {t}: {info['loss']:.6f}, diff = {np.linalg.norm(info['diff']):.6f}.")
    
    if (t % 1 == 0):
        print(f"eval at {t}")
        pi = agent.get_policy()
        print("pi", pi)
        
        n_eval = 10
        T_eval = 1000

        reward_list = []
        for rep in range(n_eval):
            reward_tot = env.eval(agent, T_eval=T_eval)
            reward_list.append(reward_tot)
        print("rewards", reward_list)

        V_pi = env.DP_pi(pi, thres=THRES)
        V_pi_avg = (V_pi*env.distr_init).sum()
        V_loss = env.V_opt_avg - V_pi_avg
        print(f"V-loss = {V_loss}")