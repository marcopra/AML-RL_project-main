"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
import os
import numpy as np
from env.custom_hopper import *
from stable_baselines3 import SAC
from CNN import *

from os import listdir
from os.path import isfile, join

DEBUG = False


def test_on(domain , model_path):

	#env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	src_model = model_path
	env_params = src_model.removesuffix('.zip').split('_')

	env_args: dict = {}

	for param in env_params:
		key, val = param.split('-')
		if val == 'False':
			env_args[key] = False
		elif val == 'True':
			env_args[key] = True
		else:
			env_args[key] = val

	env = env = my_make_env(PixelObservation= bool(env_args['img']), 
							stack_frames= int(env_args['nf']), 
							scale= bool(env_args['scaled']),
							domain= domain)
	if DEBUG:
		print('Action space:', env.action_space)
		print('State space:', env.observation_space)
		print('Dynamics parameters:', env.get_parameters())
		
	model = SAC.load(src_model)


	episodes = 50
	returns = 0
	all_rewards = []
	
	for episode in range(episodes):
		done = False
		obs = env.reset()

		episode_reward = 0

		while not done:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, info = env.step(action)
			#env.render()

			episode_reward += reward
		
		#print(f"Episode: {episode} | Return: {episode_reward}")

		all_rewards.append(episode_reward)

	if domain == 'source':
		print(f"* s-s {int(np.mean(all_rewards))}$\pm${int(np.std(all_rewards))}")
	else:
		print(f"* s-t:  {int(np.mean(all_rewards))}$\pm${int(np.std(all_rewards))}")

	

if __name__ == '__main__':
	
	src_model = []
	# path = 'trains/stacked/'
	# src_model = [f for f in listdir('trains/stacked/') if isfile(join('trains/stacked/', f))]
	# print(src_model)
	src_model.append("alg-sac_dom-source_img-False_ts-100000_nf-4_scaled-False_dr-False_batchsize-default.zip")
	src_model.append("alg-sac_dom-target_img-False_ts-100000_nf-4_scaled-False_dr-False_batchsize-default.zip")

	src_model.append("alg-sac_dom-source_img-False_ts-100000_nf-4_scaled-False_dr-True_batchsize-default.zip")

	src_model.append("alg-sac_dom-source_img-False_ts-400000_nf-4_scaled-False_dr-False_batchsize-default.zip")
	src_model.append("alg-sac_dom-target_img-False_ts-400000_nf-4_scaled-False_dr-False_batchsize-default.zip")

	src_model.append("alg-sac_dom-source_img-False_ts-400000_nf-4_scaled-False_dr-True_batchsize-default.zip")
	
	
	for src_mod in src_model:
		print("-----Testing: ", src_mod.removesuffix('.zip'), "-------------")

	# 	print("\n########################################################################## \
	# 			\t\t\t\t\t\t\t\t\t\t ON SOURCE\n\
	# ###########################################################################")
		test_on('source', src_mod)

	# 	print("\n########################################################################### \
	# 			\t\t\t\t\t\t\t\t\t\t ON TARGET\n\
	# ############################################################################")
		test_on('target', src_mod)