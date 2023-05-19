import numpy as np
from math import sin, cos, pi
from . import RMDP

# Define reward_src vectors.
reward_src_10 = np.array([0,0,-1,2,-1,0,0,-10,5,-10,0,0.9,1,0])

# reward_src_100 = np.array([sin(i*pi/100) + cos(2*i*pi/100) + sin(3*i*pi/100) for i in range(100)])
""" reward_src_100 = np.array([
    0,-10,5,-10,0,1,1,0,-1,2,-1,0,-1,-1,0,
    0,5,-10,5,0,1,1,0,1,-2,1,0,-1,-1,0,
    0,-10,5,-10,0,1,1,0,-1,2,-1,0,-1,-1,0,
    0,5,-10,5,0,1,1,0,1,-2,1,0,-1,-1,0,
    0,-10,5,-10,0,1,1,0,-1,2,-1,0,-1,-1,0,
    0,5,-10,5,0,1,1,0,1,-2,1,0,-1,-1,0,
    0,-10,5,-10,0,0,5,-10,5,0
]) """
reward_src_100 = np.array([
    0,-10,5,-10,0,1,3,0,-1,4,-1,0,-1,-1,0,
    0,3,-8,3,0,1,2,0,1,-4,1,0,-1,-2,0,
    0,-12,6,-10,0,4,1,0,-2,3,-1,0,-2,-1,0,
    0,2,-9,4,0,1,2,0,1,-5,2,0,-1,-3,0,
    0,-4,2,-8,0,3,1,0,-2,4,-5,0,-2,-1,0,
    0,3,-6,1,0,1,2,0,1,-6,1,0,-1,-4,0,
    0,-7,3,-7,0,0,2,-6,4,0
])

reward_src_1000 = np.array([sin(i*pi/1000) + cos(2*i*pi/1000) + sin(3*i*pi/1000) for i in range(1000)])


# Toy environment build functions.
def build_toy_10_env(p_perturb, beta, gamma, thres=1e-5, calc_opt=True):
    num_states  = 14
    num_actions = 3    # 0 = left, 1 = stay, 2 = right.

    
    reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
    prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            reward[s,a] = reward_src_10[s]
            
            prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
            prob[s,a,(s+a-2)%num_states] = p_perturb
            prob[s,a,(s+a)%num_states]   = p_perturb

    distr_init = np.ones(shape=(num_states,), dtype=np.float64) / num_states

    return RMDP(num_states, num_actions, distr_init, reward, prob, beta, gamma, thres, calc_opt)

def build_toy_100_env(p_perturb, beta, gamma, thres=1e-5, calc_opt=True):
    num_states  = 100
    num_actions = 3    # 0 = left, 1 = stay, 2 = right.
    
    reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
    prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            reward[s,a] = reward_src_100[s]
            
            prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
            prob[s,a,(s+a-2)%num_states] = p_perturb
            prob[s,a,(s+a)%num_states]   = p_perturb

    distr_init = np.ones(shape=(num_states,), dtype=np.float64) / num_states

    return RMDP(num_states, num_actions, distr_init, reward, prob, beta, gamma, thres, calc_opt)


def build_toy_1000_env(p_perturb, beta, gamma, thres=1e-2, calc_opt=False):
    num_states  = 1000
    num_actions = 3    # 0 = left, 1 = stay, 2 = right.

    reward = np.zeros(shape=(num_states,num_actions), dtype=np.float64)
    prob = np.zeros(shape=(num_states,num_actions,num_states), dtype=np.float64)
    for s in range(num_states):
        for a in range(num_actions):
            reward[s,a] = reward_src_1000[s]
            
            prob[s,a,(s+a-1)%num_states] = 1 - 2*p_perturb
            prob[s,a,(s+a-2)%num_states] = p_perturb
            prob[s,a,(s+a)%num_states]   = p_perturb

    distr_init = np.ones(shape=(num_states,), dtype=np.float64) / num_states

    return RMDP(num_states, num_actions, distr_init, reward, prob, beta, gamma, thres, calc_opt)