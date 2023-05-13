import numpy as np
import torch
import random
import argparse

from data import TorchDataset
from env import CartPole, Pendulum, build_small_toy_env, build_large_toy_env
from agent import RFZI_NN

seed = 20
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
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--data_size', default=int(1e6), type=int)
parser.add_argument('--batch_size', default=int(1e4), type=int)

# critic lr*
parser.add_argument('--z_func_lr', default=0.5, type=float)
parser.add_argument('--gamma', default=0.95, type=float)
parser.add_argument('--beta', default=0.01, type=float)
parser.add_argument('--tau', default=0.1, type=float)               

args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# Build environment.
if args.env == "CartPole":
    env = CartPole()
    data_path = f"./data/CartPole/CartPole_random.pkl"
elif args.env == "Pendulum":
    env = Pendulum()
    data_path = f"./data/Pendulum/Pendulum_SAC_0.3.pkl"
elif args.env == "toy_small":
    p_perturb = 0.15
    beta  = 0.01
    gamma = 0.95
    env = build_small_toy_env(p_perturb, beta, gamma)
    data_path = f"./data/Toy/toy_small_torch_random.pkl"
elif args.env == "toy_large":
    p_perturb = 0.15
    beta  = 0.01
    gamma = 0.95
    env = build_large_toy_env(p_perturb, beta, gamma)
    data_path = f"./data/Toy/toy_large_torch_random.pkl"
else:
    raise NotImplementedError

data = TorchDataset(device)
data.load(data_path)


# initialize policy
agent = RFZI_NN(env, device,
                z_func_lr=args.z_func_lr,
                gamma=args.gamma, beta=args.beta, tau=args.tau)
if args.env.startswith("toy"):
    opt_val = sum(agent.env.V_opt * agent.env.distr_init)
    print(opt_val)
    
# train RFZI
T = 2000
for t in range(T):
    info = agent.update(data, num_batches=20, batch_size=args.batch_size)
    print(f"loss {t} = {info['loss']}")

    if t % 10 == 0:
        if args.env.startswith("toy"):
            agent_actions = []
            for state in env.states:
                agent_actions.append(agent.select_action(state))
            print(f"policy = {agent_actions}")
        
        with torch.no_grad():
            rewards = []
            for t in range(args.eval_episodes):
                reward = env.eval(agent.select_action, T_eval=1000)
                rewards.append(reward)

            avg, std = np.average(rewards), np.std(rewards)
            print("---------------------------------------")
            print(f"Evaluation over {args.eval_episodes} episodes")
            print(rewards)
            if args.env.startswith("toy"):
                val = agent.calc_policy_reward()
                print('Average Reward:', val)
            print("---------------------------------------")
        
