import numpy as np
from . import RMDP

def build_toy_env(p_perturb, beta, gamma, thres=1e-5):
    num_states  = 14
    num_actions = 3    # 0 = left, 1 = stay, 2 = right.

    reward_src = np.array([0,-10,5,-10,0,1,1,0,0,0,-1,2,-1,0])
    reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
    prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            reward[s,a] = reward_src[s]
            
            prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
            prob[s,a,(s+a-2)%num_states] = p_perturb
            prob[s,a,(s+a)%num_states]   = p_perturb

    distr_init = np.ones(shape=(num_states,), dtype=np.float64) / num_states

    return RMDP(num_states, num_actions, distr_init, reward, prob, beta, gamma, thres)