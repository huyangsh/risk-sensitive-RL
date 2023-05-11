import numpy as np
import torch
import pickle as pkl

class TorchDataset:
    def __init__(self, state_dim, action_dim, max_size, device):
        # Parameters.
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.max_size   = max_size

        self.device     = device

        # Internal states.
        self.size = 0


    # Core functions: data collection.
    def start(self):
        self.state      = np.zeros((self.max_size, self.state_dim))
        self.action     = np.zeros((self.max_size, self.action_dim))
        self.next_state = np.zeros((self.max_size, self.state_dim))
        self.reward     = np.zeros((self.max_size, 1))
        self.not_done   = np.zeros((self.max_size, 1))

        self.ptr        = 0
        self.size       = 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr]        = state
        self.action[self.ptr]       = action
        self.next_state[self.ptr]   = next_state
        self.reward[self.ptr]       = reward
        self.not_done[self.ptr]     = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def finish(self):
        self.state      = torch.FloatTensor(self.state).to(self.device)
        self.action     = torch.FloatTensor(self.action).to(self.device)
        self.next_state = torch.FloatTensor(self.next_state).to(self.device)
        self.reward     = torch.FloatTensor(self.reward).to(self.device)
        self.not_done   = torch.FloatTensor(self.not_done).to(self.device)


    # Core function: data sampling.
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind], 
            self.not_done[ind]
        )


    # Core functions: loading and saving.
    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump({
                "size":         self.size,
                "state":        self.state,
                "action":       self.action,
                "next_state":   self.next_state,
                "reward":       self.reward, 
                "not_done":     self.not_done
            }, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pkl.load(f)
        
        self.size = data["size"]
        self.state = data["state"]
        self.action = data["action"]
        self.next_state = data["next_state"]
        self.reward = data["reward"]
        self.not_done = data["not_done"]