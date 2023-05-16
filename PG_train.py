import numpy as np
import random
import torch
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt
from datetime import datetime

from agent import PolicyGradientAgent
from env import RMDP, build_toy_10_env, build_toy_100_env


THRES = 1e-5
T_EST = 100

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Build environment
p_perturb = 0.15
beta  = 0.01
gamma = 0.95
env = build_toy_100_env(p_perturb, beta, gamma, THRES)

M   = 0.005
eps = 1e-2
eta = (1-gamma)**3 / (2*env.num_actions*M)
T   = int(1e6)

agent = PolicyGradientAgent(env, eta, T_EST, THRES)

pi_init = np.ones(shape=(env.num_states, env.num_actions), dtype=np.float64) / env.num_actions
agent.reset(pi_init)

eval_freq = 20
try:
    loss_list, reward_list = [], []
    for t in range(T):
        pi, info = agent.update()
        loss_list.append(info["loss"])

        print(t)
        print("V_pi", info["V_pi"])
        print("Q_pi", info["Q_pi"])
        print("pi", pi)
        print("loss", info["loss"])

        if t % eval_freq == 0:
            test_reps = 10
            test_T = 1000
            cur_rewards = []
            for rep in range(test_reps):
                cur_rewards.append( env.eval(agent.select_action, T_eval=test_T) )
            print(cur_rewards)
            reward_list.append(np.array(cur_rewards, dtype=np.float64).mean())
    
    V_pi = env.DP_pi(pi, THRES)
    loss_list.append( env.V_opt_avg - (V_pi*env.distr_init).sum() )

    
except KeyboardInterrupt:
    pass

suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.figure()
plt.plot(np.arange(0,len(loss_list)), loss_list)
plt.savefig(f"./log/active/PG/losses_{suffix}.png", dpi=200)
np.save(f"./log/active/PG/losses_{suffix}.npy", loss_list)

plt.figure()
plt.plot(np.arange(0,len(reward_list))*eval_freq, reward_list)
plt.savefig(f"./log/active/PG/rewards_{suffix}.png", dpi=200)
np.save(f"./log/active/PG/rewards_{suffix}.npy", reward_list)

test_reps = 10
test_T = 1000
reward_list = []
for rep in range(test_reps):
    reward_list.append( env.eval(agent.select_action, T_eval=test_T) )
print(reward_list)
print(np.array(reward_list, dtype=np.float64).mean())