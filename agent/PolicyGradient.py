import numpy as np
import random
from copy import deepcopy
from math import exp, log

from agent.Agent import Agent

THRES = 1e-5
EST_T = 100

class PolicyGradientAgent(Agent):
    def __init__(self, env, eta):
        self.eta   = eta
        self.env   = env

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
    
    def train(self, T, verbose=True):
        pi = np.ones(shape=(self.env.num_states, self.env.num_actions), dtype=np.float64) / self.env.num_actions

        loss_list = []
        for t in range(T):
            V_pi = self.env.DP_pi(pi, thres=THRES)
            Q_pi = self.env.V_to_Q(V_pi)

            loss = self.env.V_opt_avg - (V_pi*self.env.distr_init).sum()
            loss_list.append(loss)

            d_pi = self.env.visit_freq(pi, T=EST_T, V_pi=V_pi)[:, np.newaxis]
            grad = Q_pi * d_pi / (1-self.env.gamma)
            assert grad.shape == (self.env.num_states, self.env.num_actions)
            
            pi = pi + self.eta * grad
            for s in self.env.states:
                pi[s, :] = self._project(pi[s, :])

            if verbose:
                print(t)
                print("V_pi", V_pi)
                print("Q_pi", Q_pi)
                print("loss", loss)
                print("pi", pi)
        
        V_pi = self.env.DP_pi(pi, THRES)
        loss_list.append( self.env.V_opt_avg - (V_pi*self.env.distr_init).sum() )
        return pi, loss_list