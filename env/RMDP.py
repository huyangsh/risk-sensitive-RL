import numpy as np
import random
from copy import deepcopy
from math import exp, log
import matplotlib.pyplot as plt

THRES = 1e-5
EST_T = 100

class RMDP:
    def __init__(self, num_states, num_actions, distr_init, reward, prob, beta, gamma):
        assert distr_init.shape == (num_states,)
        assert reward.shape == (num_states, num_actions)
        assert prob.shape == (num_states, num_actions, num_states)
        assert gamma <= 1

        self.num_states  = num_states
        self.num_actions = num_actions
        self.states      = np.arange(self.num_states)
        self.actions     = np.arange(self.num_actions)
        self.distr_init  = distr_init

        self.reward = reward
        self.prob   = prob

        self.beta   = beta
        self.gamma  = gamma
        self.coeff  = gamma / beta

        self.V_opt     = self._DP_opt(thres=THRES)
        self.V_opt_avg = (self.V_opt*self.distr_init).sum()


    # Environment functions (compatible with OpenAI gym).
    def reset(self):
        self.state = random.choices(self.states, weights=self.distr_init)[0]
        return self.state
    
    def step(self, action):
        reward = self.reward[self.state, action]
        self.state = random.choices(self.states, weights=self.prob[self.state,action,:])[0]
        return self.state, reward
    
    def eval(self, pi, T, verbose=True):
        assert pi.shape == (self.num_states, self.num_actions)
        reward_tot = 0
        g_t = 1

        state = self.reset()
        if verbose: trajectory = []
        for t in range(T):
            action = random.choices(self.actions, weights=pi[state,:])[0]
            next_state, reward = self.step(action)
            if verbose: trajectory.append([state, action, reward, next_state])

            state = next_state
            reward_tot += g_t * reward
            g_t *= self.gamma
        
        if verbose:
            return trajectory, reward_tot
        else:
            return reward_tot


    # Utility: Bellman updates (using DP). 
    def _DP_opt(self, thres):
        V = np.zeros(shape=(self.num_states,), dtype=np.float64)

        diff = thres + 1
        while diff > thres:
            V_prev = V
            V = np.zeros(shape=(self.num_states,), dtype=np.float64)

            for s in self.states:
                reward_max = 0
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * exp(-self.beta * V_prev[s_])

                    reward_max = max(reward_max, self.reward[s,a] - self.coeff * log(V_pi_cum))
                
                V[s] = reward_max
            
            diff = np.linalg.norm(V - V_prev)
        
        return V

    def DP_pi(self, pi, thres):
        assert pi.shape == (self.num_states, self.num_actions)
        V = np.zeros(shape=(self.num_states,), dtype=np.float64)

        diff = thres + 1
        while diff > thres:
            V_prev = V
            V = np.zeros(shape=(self.num_states,), dtype=np.float64)

            for s in self.states:
                for a in self.actions:
                    V_pi_cum = 0
                    for s_ in self.states:
                        V_pi_cum += self.prob[s,a,s_] * exp(-self.beta * V_prev[s_])

                    V[s] += pi[s,a] * (self.reward[s,a] - self.coeff * log(V_pi_cum))
            
            diff = np.linalg.norm(V - V_prev)
        
        return V
    
    def V_to_Q(self, V):
        assert V.shape == (self.num_states,)
        Q = np.zeros(shape=(self.num_states, self.num_actions), dtype=np.float64)
        for s in self.states:
            for a in self.actions:
                V_pi_cum = 0
                for s_ in self.states:
                    V_pi_cum += self.prob[s,a,s_] * exp(-self.beta * V[s_])

                Q[s,a] = self.reward[s,a] - self.coeff * log(V_pi_cum)
        
        return Q
    

    # Utility: calculate state-visit frequency.
    def _prob_hat(self, pi, V_pi):
        if V_pi is None: V_pi = self.DP_pi(pi, thres=THRES)
        V_pi = V_pi[np.newaxis, np.newaxis, :]
        
        prob_hat = self.prob * np.exp(-self.beta*V_pi)
        prob_hat /= prob_hat.sum(axis=2, keepdims=True)
        return prob_hat
    
    def _transit(self, distr, prob, pi):
        distr_new = np.zeros(shape=(self.num_states,), dtype=np.float64)
        for s in self.states:
            for a in self.actions:
                for s_ in self.states:
                    distr_new[s_] += distr[s] * pi[s,a] * prob[s,a,s_]
        
        return distr_new

    def visit_freq(self, pi, T, V_pi=None):
        assert pi.shape == (self.num_states, self.num_actions)
        prob_hat = self._prob_hat(pi, V_pi)

        distr_cur = deepcopy(self.distr_init)
        g_t = 1
        d_pi = distr_cur
        for t in range(T):
            g_t *= self.gamma
            distr_cur = self._transit(distr_cur, prob_hat, pi)
            d_pi += g_t * distr_cur
        
        d_pi *= 1-self.gamma
        return d_pi