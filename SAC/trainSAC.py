"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse

from env.custom_hopper import *

from stable_baselines3 import SAC,PPO

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n-episodes', default=100, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--domain', default='source', type=str, help='Choose train domain')
	parser.add_argument('--pixel_obs', default=True, type=bool, help='Activate pixel observation')
	parser.add_argument('--n_frames', default=4, type=int, help='Number of stacked frames')
	parser.add_argument('--scaled_frames', default=False, type=bool, help='Activate to use scaled frames')
	parser.add_argument('--algorithm', default='sac', type=str, help='choose the algorithm [ppo, sac]')
	
	
	return parser.parse_args()

args = parse_args()

def main():

	# # env = gym.make('CustomHopper-target-v0')
	# env = my_make_env(PixelObservation=args.pixel_obs, stack_frames=args.n_frames, scale=args.scaled_frames, domain=args.domain)
	env = PixelObservationWrapper(gym.make('CustomHopper-source-v0'))
	   
	# env = env['pixels']


	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	"""
        TODO:

            - train a policy with stable-baselines3 on source env
            - test the policy with stable-baselines3 on <source,target> envs
   
		Training
	"""
	if args.algorithm == 'sac':
		if args.pixel_obs:
			model = SAC("MultiInputPolicy", env, verbose=1)
		else:
			model = SAC("MultiInputPolicy", env, verbose=1)
	elif args.algorithm == 'ppo':
		if args.pixel_obs:
			model = PPO("MlpPolicy", env, verbose=1)
		else:
			model = PPO("MlpPolicy", env, verbose=1)

	# model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=200000, log_interval=10)
	model.save(f"{args.algorithm}_{args.domain}_{'image' if args.pixel_obs else 'normal'}")
	

	

if __name__ == '__main__':
	main()