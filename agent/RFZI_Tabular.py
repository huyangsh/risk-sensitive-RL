import numpy as np
import random
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt

class RFZI_Tabular:
    def __init__(self, env):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.reward = env.reward
        self.beta = env.beta
        self.gamma = env.gamma
        
        # Internal state
        self.Z = None

    def reset(self, Z_init):
        assert Z_init.shape == (self.num_states, self.num_actions)
        self.Z = Z_init

    def update(self, dataset, verbose=True):
        num_data = len(dataset)

        Z_next = np.zeros_like(self.Z)

        # Calculate best response.
        Z_max = np.exp( (self.reward - np.log(self.Z)/self.beta).max(axis=1) * (-self.beta*self.gamma) )

        for s,a,r,s_ in dataset:
            s, a, s_ = int(s), int(a), int(s_)
            Z_next[s,a] += Z_max[s_]
        Z_next /= num_data

        info = {}
        if verbose:
            # Test empirical loss.
            loss = 0
            for s,a,r,s_ in dataset:
                s, a, s_ = int(s), int(a), int(s_)
                loss += (Z_next[s,a] - Z_max[s_]) ** 2
            loss /= num_data

            info = {
                "loss": loss,
                "diff": self.Z - Z_next
            }

        self.Z = Z_next
        return self.Z, info

    def get_policy(self):
        a_max = (self.reward - self.Z/self.beta).argmax(axis=1)
        pi = np.zeros(shape=(self.num_states,self.num_actions), dtype=np.float64)
        pi[np.arange(self.num_states), a_max] = 1
        return pi