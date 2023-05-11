import numpy as np
import torch
import argparse

from data import TorchDataset
from env import CartPolePerturbed
from agent import RFZI_NN

parser = argparse.ArgumentParser()
# epsilon used to generate data
parser.add_argument('--data_eps', default=0.3, type=float)
# policy used to generate data
parser.add_argument('--gendata_pol', default='ppo', type=str)  
parser.add_argument('--env', default='CartPole-v0', type=str)
parser.add_argument('--max_trn_steps', default=int(5e5), type=float)
parser.add_argument('--eval_freq', default=10, type=float)
parser.add_argument('--eval_episodes', default=10, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--data_size', default=int(1e6), type=int)
parser.add_argument('--batch_size', default=int(1e3), type=int)

# critic lr*
parser.add_argument('--z_func_lr', default=0.1, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--tau', default=0.5, type=float)               

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Build environment.
env = CartPolePerturbed()

state_dim = 4
action_dim = 1
num_actions = 2

data_path = f"./data/CartPole/{args.env}_{args.gendata_pol}_e{args.data_eps}"
data = TorchDataset(state_dim, action_dim, args.device)
data.load(data_path, args.data_size)


# initialize policy
agent = RFZI_NN(state_dim, action_dim, num_actions, args.device,
                z_func_lr=args.z_func_lr,
                gamma=args.gamma, beta=args.beta, tau=args.tau, env=env)
    
# train RFZI
T = 1000
for t in range(T):
    info = agent.update(data, batch_size=args.batch_size)
    print(f"loss {t} = {info['loss']:.6f}")

    if t % 10 == 0:
        with torch.no_grad():
            avg, std = env.eval(agent, args.eval_episodes)
        
