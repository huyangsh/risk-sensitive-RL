import numpy as np
import torch
import random
import argparse

from data import TorchDataset
from env import CartPole, Pendulum
from agent import RFZI_NN

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
# epsilon used to generate data
parser.add_argument('--data_eps', default=0.3, type=float)
# policy used to generate data
parser.add_argument('--gendata_pol', default='ppo', type=str)  
parser.add_argument('--env', default='CartPole', type=str)
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
parser.add_argument('--tau', default=1.0, type=float)               

args = parser.parse_args()
print(args)

# Build environment.

if args.env == "CartPole":
    env = CartPole()
    data_path = f"./data/CartPole/CartPole_PPO_0.3.pkl"
elif args.env == "Pendulum":
    env = Pendulum()
    data_path = f"./data/Pendulum/Pendulum_random.pkl"
else:
    raise NotImplementedError

data = TorchDataset(args.device)
data.load(data_path)


# initialize policy
agent = RFZI_NN(env, args.device,
                z_func_lr=args.z_func_lr,
                gamma=args.gamma, beta=args.beta, tau=args.tau)
    
# train RFZI
T = 1000
for t in range(T):
    info = agent.update(data, num_batches=100, batch_size=args.batch_size)
    print(f"loss {t} = {info['loss']}")

    if t % 10 == 0:
        with torch.no_grad():
            rewards = []
            for t in range(args.eval_episodes):
                reward = env.eval(agent, T_eval=1000)
                rewards.append(reward)

            avg, std = np.average(rewards), np.std(rewards)
            print("---------------------------------------")
            print(f"Evaluation over {args.eval_episodes} episodes")
            print(rewards)
            print("---------------------------------------")
        
