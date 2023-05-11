import numpy as np
import random
from copy import deepcopy
from math import exp, log

from agent.Agent import Agent

THRES = 1e-5
EST_T = 100

class PolicyGradientAgent(Agent):
    def __init__(self, env, eta):
        self.eta = eta
        self.env = env
        
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.states = env.states
        self.actions = env.actions

        self.reward = env.reward
        self.beta = env.beta
        self.gamma = env.gamma

    def _l2_project(self, r, a, b):
        # Implements l2-projection onto the simplex:
        #   min_y  ||r-x||^2
        #   s.t.   1^T x = 1
        #          a <= x <= b
        # REF: http://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html.
        assert (a.sum()<=1) and (b.sum()>=1) and np.all(a<=b), "Error: projection infeasible."
        lambdas = np.append(a-r, b-r)
        idx = np.argsort(lambdas)
        lambdas = lambdas[idx]
        active = np.cumsum((idx < r.size)*2 - 1)[:-1]
        diffs = np.diff(lambdas, n=1)
        totals = a.sum() + np.cumsum(active*diffs)
        i = np.searchsorted(totals, 1.0)
        lam = (1 - totals[i]) / active[i] + lambdas[i+1]
        return np.clip(r + lam, a, b)

    def _project(self, x):
        assert x.shape == (self.env.num_actions,)
        return self._l2_project(x, np.zeros_like(x), np.ones_like(x))
    
    def reset(self, pi_init):
        assert pi_init.shape == (self.num_states, self.num_actions)
        self.pi = pi_init
        
        return self.pi
    
    def update(self):
        V_pi = self.env.DP_pi(self.pi, thres=THRES)
        Q_pi = self.env.V_to_Q(V_pi)
        loss = self.env.V_opt_avg - (V_pi*self.env.distr_init).sum()

        d_pi = self.env.visit_freq(self.pi, T=EST_T, V_pi=V_pi)[:, np.newaxis]
        grad = Q_pi * d_pi / (1-self.env.gamma)
        assert grad.shape == (self.num_states, self.num_actions)
        
        self.pi = self.pi + self.eta * grad
        for s in self.env.states:
            self.pi[s, :] = self._project(self.pi[s, :])
        
        return self.pi, {"loss": loss, "V_pi": V_pi, "Q_pi": Q_pi}