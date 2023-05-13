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
        # Warning: use sin-cos representation!
        # self.l1 = nn.Linear(dim_state + dim_action, 32)
        self.l1 = nn.Linear(16*dim_state + dim_action, 32)
        self.l2 = nn.Linear(32, 8)
        self.l3 = nn.Linear(8, 1)
    
    def _state_embedding(self, state, action):
        # return torch.cat([state, action], 1)
        return torch.cat([
            torch.sin(state), torch.cos(state),
            torch.sin(2*state), torch.cos(2*state),
            torch.sin(3*state), torch.cos(3*state),
            torch.sin(4*state), torch.cos(4*state),
            torch.sin(5*state), torch.cos(5*state),
            torch.sin(6*state), torch.cos(6*state),
            torch.sin(7*state), torch.cos(7*state),
            torch.sin(8*state), torch.cos(8*state),
            action
        ], 1)

    def forward(self, state, action):
        # Warning: use sin-cos representation!
        # z = F.relu(self.l1(torch.cat([state, action], 1)))
        z = F.relu(self.l1(self._state_embedding(state, action)))
        z = F.relu(self.l2(z))
        z = F.sigmoid(self.l3(z))
        return z
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device="cpu"):
        self.to(device)
        self.load_state_dict(
            torch.load(path, map_location=torch.device(device))
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
        if len(self.actions_tensor.shape) == 1: self.actions_tensor = self.actions_tensor[:, None]
        self.actions_tensor = self.actions_tensor.to(device)

        if type(env.reward) == np.ndarray:
            self.reward     = lambda s,a: env.reward[int(s),int(a)]
        else:
            self.reward     = env.reward
        
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
            next_states_np = next_states.cpu().numpy()
            next_rewards = np.zeros(shape=(batch_size, self.num_actions), dtype=np.float32)
            for i in range(batch_size):
                s_ = next_states_np[i]  # Warning: the tabular case needs [0]! Try to solve this.
                for j in range(self.num_actions):
                    a_ = self.actions[j]
                    next_rewards[i, j] = self.reward(s_, a_)
            next_rewards = torch.FloatTensor(next_rewards).flatten().to(self.device)
            
            # Calculating the loss w.r.t. target Z_function.
            with torch.no_grad():
                next_states = torch.repeat_interleave(next_states, self.num_actions, dim=0)
                
                target_Z = self.z_func_target(next_states, test_actions).flatten()
                target_Z = self.beta*next_rewards - self.gamma*torch.log(target_Z)
                target_Z = target_Z.reshape(shape=(batch_size, self.num_actions)).amax(dim=1)
                target_Z = torch.exp(-target_Z)

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

            if type(state) in (int, np.int32, np.int64): state = [state]
            state_tensor = torch.FloatTensor(state).repeat(repeats=(self.num_actions, 1)).to(self.device)
            rewards = rewards - 1/self.beta * np.log(self.z_func_target(state_tensor, self.actions_tensor).cpu().detach().flatten().numpy())
        
        return self.actions[rewards.argmax()]

    def load(self, path):
        self.z_func_current.load(path)
        self.z_func_target.load(path)
    
    def save(self, path):
        self.z_func_target.save(path)


    # Help functions.
    def calc_Tz(self, Z1):
        Z = np.zeros([self.env.num_states, self.env.num_actions])
        for s in self.env.states:
            for a in self.env.actions:
                for s_ in self.env.states:
                    temp = -10000
                    for a_ in self.env.actions:
                        temp_ = self.beta * self.env.reward[s_, a_] - np.log(Z1[s_,a_])#np.log(self.z_func_target(torch.tensor([[float(s_)]]),
                                                                                         #torch.tensor([[float(a_)]]))[0][0].detach().numpy())
                        temp = max(temp, temp_)
                    Z[s, a] += self.env.prob[s, a, s_] * np.exp(-self.gamma * temp)
        return Z

    def calc_Z(self,net):
        Z = np.zeros([self.env.num_states, self.env.num_actions])
        for s in self.env.states:
            for a in self.env.actions:
                Z[s,a] = net(torch.tensor([[float(s)]]).to(self.device), torch.tensor([[float(a)]]).to(self.device))[0][0].detach().cpu().numpy()
        return Z

    def calc_Q(self,net):
        Q = np.zeros([self.env.num_states, self.env.num_actions])
        for s in self.env.states:
            for a in self.env.actions:
                Q[s,a] = self.env.reward[s,a] - \
                1/self.beta*np.log(net(torch.tensor([[float(s)]]).to(self.device), torch.tensor([[float(a)]]).to(self.device))[0][0].detach().cpu().numpy())
                # 1/self.beta*np.log(min(1,max(np.exp(-self.beta/(1-self.gamma)),net(torch.tensor([[float(s)]]), torch.tensor([[float(a)]]))[0][0].detach().numpy())))

        return Q

    def calc_err(self):
        Z1 = self.calc_Z(self.z_func_target)
        Z1 = self.calc_Tz(Z1)
        Z2 = self.calc_Z(self.z_func_current)
        return np.linalg.norm(Z1 - Z2)

    def calc_opt(self):
        return self.env.V_to_Q(self.env.V_opt)

    def calc_loss(self, Z_current, Z_target, data, batchsize=10000):
        loss = 0
        states, actions, _, next_states, _ = data.sample(batchsize)
        for i in range(batchsize):
            s = states[i][0].cpu().numpy().astype(int)
            a = actions[i][0].cpu().numpy().astype(int)
            s_ = next_states[i][0].cpu().numpy().astype(int)
            target = Z_target[s_,:]
            target = self.beta * self.env.reward[s_,:] - np.log(target)
            target = np.max(target)
            target = np.exp(-self.gamma * target)

            current = Z_current[s,a]

            loss += (current - target)**2

        return loss/batchsize

    def calc_pi(self):
        pi = np.zeros([self.env.num_states, self.env.num_actions])
        Q = self.calc_Q(self.z_func_current)
        for s in self.env.states:
            idx = np.argmax(Q[s,:])
            pi[s,idx] = 1
        return pi

    def calc_policy_reward(self):
        V = self.env.DP_pi(self.calc_pi(), 1e-5)
        return np.sum(V*self.env.distr_init)