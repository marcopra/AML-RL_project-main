"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse

from env.custom_hopper import *

from stable_baselines3 import SAC,PPO
try:
	from alexnet import policy_kwargs
except:
	pass

# RUN TODO with DR:
# ts = 200k fixed (all following 200k steps)
#(dr = True attivare a mano) See custom_hopper row 100
# 1. --n_frames 1 
# 2. --n_f 1 -sf
# 3. --n_f 4 
# 4. --n_f 4 -sf


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--time_steps', '-ts', default=200_000, type=int, help='Number of training episodes')
	parser.add_argument('--print-every', default=500, type=int, help='Print info every <> episodes')
	parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
	parser.add_argument('--domain', default='source', type=str, help='Choose train domain')
	parser.add_argument('--pixel_obs', default=False, type=bool, help='Activate pixel observation')
	parser.add_argument('--n_frames','-nf' , default=4, type=int, help='Number of stacked frames')
	parser.add_argument('--scaled_frames', '-sf', action='store_true', default=False, help='Activate to use scaled frames')
	parser.add_argument('--algorithm', default='sac', type=str, help='choose the algorithm [ppo, sac]')
	
	return parser.parse_args()

args = parse_args()

def main():

	env = my_make_env(PixelObservation=args.pixel_obs, stack_frames=args.n_frames, scale=args.scaled_frames, domain=args.domain)
	

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
			model = SAC("CnnPolicy", 
						env = env, 
						buffer_size=5000,
						batch_size=8,
						policy_kwargs=policy_kwargs, 
						verbose=1,
						device=args.device,
						tensorboard_log="./Hopper_CNN/"
						)
		else:
			model = SAC("MlpPolicy", 
						env = env, 
						buffer_size=5000,
						verbose=1,
						device=args.device,
						tensorboard_log="./Normal_Hopper_CNN/")

	elif args.algorithm == 'ppo':
		if args.pixel_obs:
			model = PPO("CnnPolicy", 
						env = env, 
						buffer_size=5000,
						batch_size=8,
						policy_kwargs=policy_kwargs, 
						verbose=1,
						device=args.device,
						tensorboard_log="./PPO_Hopper_CNN/"
						)
		else:
			model = PPO("MlpPolicy", env, verbose=1,
						device=args.device)

	# tensorboard --logdir ./Normal_Hopper_CNN/
	model.learn(total_timesteps=args.time_steps, log_interval=100, progress_bar=True, tb_log_name=f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	model.save(f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	
	model.learn(total_timesteps=args.time_steps, log_interval=100, progress_bar=True, tb_log_name=f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{2*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	model.save(f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{2*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	
	model.learn(total_timesteps=args.time_steps, log_interval=100, progress_bar=True, tb_log_name=f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{3*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	model.save(f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{3*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	
	model.learn(total_timesteps=args.time_steps, log_interval=100, progress_bar=True, tb_log_name=f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{4*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	model.save(f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{4*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	
	model.learn(total_timesteps=args.time_steps, log_interval=100, progress_bar=True, tb_log_name=f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{5*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	model.save(f"alg-{args.algorithm}_dom-{args.domain}_img-{args.pixel_obs}_ts-{5*args.time_steps}_nf-{args.n_frames}_scaled-{args.scaled_frames}")
	

if __name__ == '__main__':
	main()