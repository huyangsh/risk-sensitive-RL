import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import math


class Z_Func(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Z_Func, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 32)
        self.l2 = nn.Linear(32, 8)
        self.l3 = nn.Linear(8, 1)

        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)

    def forward(self, state, action):
        z = F.relu(self.l1(torch.cat([state, action], 1)))
        z = F.relu(self.l2(z))
        z = F.sigmoid(self.l3(z))
        return z
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, 
                                        map_location=torch.device(device))) 


class RFZI_NN:
    def __init__(self, state_dim, action_dim, num_actions, device, 
                 z_func_lr=1e-3, gamma=0.99, beta=0.01, tau=0.005, env=None):
        self.env = env
        self.reward = env.reward
        
        # learning rates*
        self.z_func_lr = z_func_lr
        latent_dim = action_dim * 2
        
        # initialize
        self.z_func_current = Z_Func(state_dim, action_dim).to(device)
        self.z_func_target = copy.deepcopy(self.z_func_current)
        self.z_func_optimizer = torch.optim.SGD(self.z_func_current.parameters(),
                                                lr=z_func_lr)  #eps=self.adam_eps,
                                                
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        self.device = device
        
        self.eps = 1e-7

        

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.available_actions = np.arange(num_actions)

    def select_action(self, state):
        # WARNING: this is not compatible with continuous actions!
        state_torch = torch.FloatTensor(state).repeat((len(self.available_actions), 1)).to(self.device)
        actions_torch = torch.FloatTensor(self.available_actions)[:, None]

        rewards = np.zeros(shape=(len(self.available_actions),), dtype=np.float32)
        for j in range(len(self.available_actions)):
            a = self.available_actions[j]
            rewards[j] = self.reward(state, a)
        rewards = rewards - 1/self.beta * self.z_func_target(state_torch, actions_torch).cpu().detach().flatten().numpy()
        return rewards.argmax()

    def update(self, data, batch_size=100):
        for b in range(100):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = data.sample(batch_size)

            # Get rewards.
            next_rewards = np.zeros(shape=(batch_size, len(self.available_actions)), dtype=np.float32)
            for i in range(batch_size):
                s_ = next_state[i, :]
                for j in range(len(self.available_actions)):
                    a_ = self.available_actions[j]
                    next_rewards[i, j] = self.reward(s_, a_)
            next_rewards = torch.FloatTensor(next_rewards).flatten().to(self.device)
            
            # Z_Func Training
            with torch.no_grad():
                # Compute value of perturbed actions sampled from the VAE
                next_state = torch.repeat_interleave(next_state, len(self.available_actions), 0)
                test_actions = torch.FloatTensor(self.available_actions).repeat(batch_size)[:, None]
                target_Z = self.z_func_target(next_state, test_actions).flatten()
                # print("Z:", target_Z.min(), target_Z.max())
                target_Z = self.beta*next_rewards - torch.log(target_Z)
                target_Z = target_Z.reshape(shape=(batch_size, len(self.available_actions))).amax(dim=1)
                target_Z = torch.exp(-self.gamma * target_Z)

            current_Z = self.z_func_current(state, action).flatten()
            z_func_loss = F.mse_loss(current_Z, target_Z)

            self.z_func_optimizer.zero_grad()
            z_func_loss.backward()
            """print(i, j, torch.norm(self.z_func_current.l1.weight.grad),
                    torch.norm(self.z_func_current.l2.weight.grad),
                    torch.norm(self.z_func_current.l3.weight.grad))
            print("weight", torch.norm(self.z_func_current.l1.weight.data),
                    torch.norm(self.z_func_current.l2.weight.data),
                    torch.norm(self.z_func_current.l3.weight.data))"""
            self.z_func_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.z_func_current.parameters(), self.z_func_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"loss": z_func_loss}