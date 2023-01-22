"""
Train an RL agent on the OpenAI Gym Hopper environment
"""

import torch
import argparse

from env.custom_hopper import *

from stable_baselines3 import SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from alexnet import policy_kwargs
from resnet18 import policy_kwargs_renset18
torch.cuda.empty_cache()


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--time_steps', default=100000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=500, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--domain', default='source', type=str, help='Choose train domain')
	parser.add_argument('--pixel_obs', default=False, type=bool, help='Activate pixel observation')
	parser.add_argument('--n_frames','-nf' , default=4, type=int, help='Number of stacked frames')
	parser.add_argument('--scaled_frames', '-sf', action='store_true', default=False, help='Activate to use scaled frames')
	parser.add_argument('--algorithm', default='sac', type=str, help='choose the algorithm [ppo, sac]')
	
	return parser.parse_args()


def main():

	args = parse_args()

	env = my_make_env(PixelObservation=args.pixel_obs, stack_frames=args.n_frames, scale=args.scaled_frames, domain=args.domain)
	env.render(mode="rgb_array", width=84, height=84)

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	if args.algorithm == 'sac':
		if args.pixel_obs:
			
			mean_noise = np.array([0, 0, 0])
			sigma_noise = np.array([0.2, 0.2, 0.2])
			model = SAC("CnnPolicy", 
						action_noise= OrnsteinUhlenbeckActionNoise(mean= mean_noise,  sigma= sigma_noise),
						env = env, 
						buffer_size= 1_000_000, 
						policy_kwargs=policy_kwargs_renset18, 
						verbose=1,
						device = 'cuda:0',
						tensorboard_log="./Hopper_CNN/"
						)
		else:
			model = SAC("MlpPolicy", 
						env = env, 
						buffer_size=5000,
						verbose=1,
						device=args.device,
						tensorboard_log="./Normal_Hopper_CNN/")
	
	model.learn(total_timesteps=args.time_steps, log_interval=100, progress_bar=True, tb_log_name=f" Resnet-{args.time_steps}-test") 
	model.save(f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")

if __name__ == '__main__':
	main()