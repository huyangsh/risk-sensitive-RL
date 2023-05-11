import numpy as np
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Agent


class Z_Func(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Z_Func, self).__init__()
        self.l1 = nn.Linear(dim_state + dim_action, 32)
        self.l2 = nn.Linear(32, 8)
        self.l3 = nn.Linear(8, 1)

    def forward(self, state, action):
        z = F.relu(self.l1(torch.cat([state, action], 1)))
        z = F.relu(self.l2(z))
        z = F.sigmoid(self.l3(z))
        return z
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device="cpu"):
        self.to(device)
        self.load_state_dict(
            torch.load(filename, map_location=torch.device(device))
        ) 


class RFZI_NN(Agent):
    def __init__(self, env, device, 
                 z_func_lr=1e-3, gamma=0.99, beta=0.01, tau=0.005, ):
        # Environment information.
        self.env            = env

        self.dim_state      = env.dim_state
        self.dim_action     = env.dim_action
        self.num_actions    = env.num_actions
        self.actions        = env.actions
        self.actions_tensor = torch.FloatTensor(np.array(self.actions))

        self.reward         = env.reward
        self.beta           = beta
        self.gamma          = gamma
        
        # Learning parameters.
        self.z_func_lr  = z_func_lr
        self.tau        = tau
        self.eps        = 1e-7
        
        # Internal states.
        self.device = device
        self.z_func_current = Z_Func(self.dim_state, self.dim_action).to(device)
        self.z_func_optimizer = torch.optim.SGD(
            self.z_func_current.parameters(),
            lr=z_func_lr
        )
        self.reset()
        

    # Core functions.
    def reset(self):
        nn.init.xavier_uniform_(self.z_func_current.l1.weight)
        nn.init.xavier_uniform_(self.z_func_current.l2.weight)
        nn.init.xavier_uniform_(self.z_func_current.l3.weight)

        self.z_func_target = copy.deepcopy(self.z_func_current)

    def update(self, dataset, num_batches, batch_size):
        test_actions = self.actions_tensor.repeat((batch_size, 1))

        loss_list = []
        for b in range(num_batches):
            # Sample a batch from dataset.
            states, actions, _, next_states, _ = dataset.sample(batch_size)

            # Get rewards.
            next_rewards = np.zeros(shape=(batch_size, self.num_actions), dtype=np.float32)
            for i in range(batch_size):
                s_ = next_states[i]
                for j in range(self.num_actions):
                    a_ = self.actions[j]
                    next_rewards[i, j] = self.reward(s_, a_)
            next_rewards = torch.FloatTensor(next_rewards).flatten().to(self.device)
            
            # Calculating the loss w.r.t. target Z_function.
            with torch.no_grad():
                next_states = torch.repeat_interleave(next_states, self.num_actions, dim=0)
                
                target_Z = self.z_func_target(next_states, test_actions).flatten()
                target_Z = self.beta*next_rewards - torch.log(target_Z)
                target_Z = target_Z.reshape(shape=(batch_size, self.num_actions)).amax(dim=1)
                target_Z = torch.exp(-self.gamma*target_Z)

            current_Z = self.z_func_current(states, actions).flatten()
            z_func_loss = F.mse_loss(current_Z, target_Z)
            loss_list.append(z_func_loss.item())

            # Batch updates.
            self.z_func_optimizer.zero_grad()
            z_func_loss.backward()
            self.z_func_optimizer.step()

        # Update the target network after all batches.
        for param, target_param in zip(self.z_func_current.parameters(), self.z_func_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"loss": loss_list}
    
    def select_action(self, state):
        with torch.no_grad():
            rewards = np.array([self.reward(state, a) for a in self.actions], dtype=np.float32)

            state_tensor = torch.FloatTensor(state).repeat(repeats=(self.num_actions, 1)).to(self.device)
            rewards = rewards - 1/self.beta * self.z_func_target(state_tensor, self.actions_tensor).cpu().detach().flatten().numpy()
        
        return self.actions[rewards.argmax()]

    def load(self, filename):
        self.z_func_current.load(filename)
        self.z_func_target.load(filename)
    
    def save(self, filename):
        self.z_func_target.save(filename)