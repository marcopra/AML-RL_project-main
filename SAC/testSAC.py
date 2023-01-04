"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
import os
import numpy as np
from env.custom_hopper import *
from stable_baselines3 import SAC

def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	

	model = SAC.load("SAC_source_model_no_dr")


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
			env.render()

			episode_reward += reward
		
		print(f"Episode: {episode} | Return: {episode_reward}")

		all_rewards.append(episode_reward)


	print(f"AvgReward: {np.mean(all_rewards)} and devSTD: {np.std(all_rewards)}")
	


if __name__ == '__main__':
	main()