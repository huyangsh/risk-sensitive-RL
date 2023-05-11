import numpy as np
import random
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt

from agent import PolicyGradientAgent
from env import RMDP, build_toy_env


THRES = 1e-5
T_EST = 100

seed = 0
random.seed(seed)
np.random.seed(seed)

# Build environment
p_perturb = 0.15
beta  = 0.01
gamma = 0.95
env = build_toy_env(p_perturb, beta, gamma, THRES)

M   = 0.005
eps = 1e-2
eta = (1-gamma)**3 / (2*env.num_actions*M)
T   = int(16*env.num_actions*M**4 / (((1-gamma)**4) * (eps**2))) * 100

agent = PolicyGradientAgent(env, eta, T_EST, THRES)

pi_init = np.ones(shape=(env.num_states, env.num_actions), dtype=np.float64) / env.num_actions
agent.reset(pi_init)

loss_list = []
for t in range(T):
    pi, info = agent.update()
    loss_list.append(info["loss"])

    print(t)
    print("V_pi", info["V_pi"])
    print("Q_pi", info["Q_pi"])
    print("loss", info["loss"])
    print("pi", pi)

V_pi = env.DP_pi(pi, THRES)
loss_list.append( env.V_opt_avg - (V_pi*env.distr_init).sum() )

plt.plot(np.arange(0,T+1), loss_list)
plt.savefig("./fig.png", dpi=200)

test_reps = 10
test_T = 1000
reward_list = []
for rep in range(test_reps):
    reward_list.append( env.eval(pi, T=test_T) )
print(reward_list)
print(np.array(reward_list, dtype=np.float64).mean())