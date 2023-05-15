import argparse
import numpy as np
import torch
import pickle as pkl
import random

from agent import RFQI, RFZI_NN
from env import CartPole

seed = 1024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

agent_type = "RFZI_NN"
eval_episodes = 20
env = CartPole(sigma=0.001, T_max=200)
device = torch.device("cpu")

if agent_type == "RFQI":
    agent_path = f"./log/baseline/RFQI_CartPole"

    agent = RFQI(env.dim_state, env.dim_action, 0, 1, device, "discrete",
                 gamma=0.99, tau=0.005, lmbda=0.75, phi=0.1)
    agent.actor.load(f'{agent_path}_actor', device=device)
    agent.critic.load(f'{agent_path}_critic', device=device)
    agent.vae.load(f'{agent_path}_vae', device=device)

    select_action = lambda state: np.rint(agent.select_action(np.array(state)))
elif agent_type == "RFZI_NN":
    agent_path = f"./log/active/CartPole_0.0_100000_10000_0.5_1.0_20230514_161904"

    with open(agent_path + ".pkl", "rb") as f:
        settings = pkl.load(f)
    
    def emb_func(state):
        return torch.cat([state, torch.sin(state[:,2][:,None]), torch.cos(state[:,2][:,None])], 1)
    dim_emb = env.dim_state + 2

    agent = RFZI_NN(
        env=env, device=device,
        beta=settings.beta, gamma=settings.gamma, 
        lr=0, tau=0,
        emb_func=emb_func, dim_emb=dim_emb,
        dim_hidden=(256*env.dim_state, 32)
    )
    agent.load(agent_path + ".ckpt")

    select_action = agent.select_action
else:
    raise NotImplementedError



def perturb_parameter(title, ps, select_action, agent_type, env, reset_func):
    print("---------------------------------------")
    print(f"test title: {title}")
    avgs, stds = [], []
    for p in ps:
        env.reset()
        force_mag = env.force_mag * (1 + p)
        rewards = []
        for _ in range(eval_episodes):
            state, done = reset_func(env, force_mag), False
            episode_reward = 0.0
            while not done:
                action = select_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        
        avg_reward = np.sum(rewards) / eval_episodes
        
        print(f" p = {p:.4f}: avg = {avg_reward:.4f}.")
        
        avgs.append(avg_reward)
        stds.append(np.std(rewards))
    
    np.save(f'./log/perturb/{agent_type}_{title}_avgs.npy', avgs)
    np.save(f'./log/perturb/{agent_type}_{title}_stds.npy', stds)
    print("---------------------------------------")



# Force magnitude
ps_fm = np.arange(-0.8, 4.0, 0.4)
reset_func = lambda env, x: env.reset(force_mag=x, init_angle_mag=0.20)
perturb_parameter("force-mag", ps_fm, select_action, agent_type, env, reset_func)

# Gravity
ps_g = np.arange(-3.0, 3.0, 0.5)
reset_func = lambda env, x: env.reset(gravity=x, init_angle_mag=0.20)
perturb_parameter("gravity", ps_g, select_action, agent_type, env, reset_func)

# Pole length
ps_len = np.arange(-0.8, 4.0, 0.4)
reset_func = lambda env, x: env.reset(length=x, init_angle_mag=0.20)
perturb_parameter("length", ps_len, select_action, agent_type, env, reset_func)