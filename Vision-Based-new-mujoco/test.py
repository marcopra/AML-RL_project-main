"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
from env_utils import *
import gymnasium as gym
import argparse
import os
from Model.Actor import Policy
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from utils import obs_processing


torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models_saved/model10000.pkl", type=str, help='Model path')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=25, type=int, help='Number of test episodes')
    parser.add_argument('--mode', default="stacked_frames", type=str, help='Mode of testing [single, stacked_frames]')

    return parser.parse_args()

args = parse_args()


def main():

    MODE = args.mode
	# env = gym.make('CustomHopper-source-v0')
    if MODE == "stacked_frames":
        env = my_make_env()
    elif MODE == "single":
        env = PixelObservationWrapper(make_env(domain="source", render_mode='rgb_array'))
        
    else:
        assert False, f"Da mettere descrizione"

    # print('Action space:', env.action_space)
    # print('State space:', env.observation_space)
    # print('Dynamics parameters:', env.get_parameters())
	

    s, _ = env.reset()
    if MODE == "stacked_frames":
        state_dim = s.shape
    elif MODE == "single":
        state_dim = obs_processing(s['pixels']).shape
        
    
    action_dim = env.action_space.shape[0]
    
    policy = Policy(state_dim, action_dim).to(device=args.device)

    policy.load_state_dict(torch.load(args.model), strict=True)
    
    total_reward = 0
    
    for episode in range(args.episodes):
        episode_reward = 0
        done = False
        state, _ = env.reset()
        while not done:

            if MODE == "stacked_frames":
                action = policy.get_action(state)
            elif MODE == "single":
                action = policy.get_action(obs_processing(state['pixels']))
        
            
            state, reward, done, _  = env.step(action=action)
            env.render()

            episode_reward += reward
    
        print(f"Episode: {episode} | Return: {episode_reward}")
        total_reward += episode_reward
    
    print(f"Average Reward = {total_reward/args.episodes}")
    
	

if __name__ == '__main__':
	main()