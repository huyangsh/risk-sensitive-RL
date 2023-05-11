import numpy as np
import random
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt

from agent import PolicyGradientAgent
from env import RMDP


THRES = 1e-5

# Build environment.
num_states  = 14
num_actions = 3    # 0 = left, 1 = stay, 2 = right.

p_perturb = 0.15
reward_src = np.array([0,-10,5,-10,0,1,1,0, 0,0,-1,2,-1,0])
reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
for s in range(num_states):
    for a in range(num_actions):
        reward[s,a] = reward_src[s]
        
        prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
        prob[s,a,(s+a-2)%num_states] = p_perturb
        prob[s,a,(s+a)%num_states]   = p_perturb

distr_init = np.ones(shape=(num_states,), dtype=np.float64) / num_states

beta  = 0.01
gamma = 0.95
env = RMDP(num_states, num_actions, distr_init, reward, prob, beta, gamma)

M   = 0.005
eps = 1e-2
eta = (1-gamma)**3 / (2*num_actions*M)
T   = int(16*num_actions*M**4 / (((1-gamma)**4) * (eps**2))) * 100

agent = PolicyGradientAgent(env, eta)

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