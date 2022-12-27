"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
import os
from env.custom_hopper import *
from stable_baselines3 import PPO


def main():

	#env = gym.make('CustomHopper-source-v0')
	env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	

	model = PPO.load("my_model")

	episodes = 30
	returns = 0
	
	for episode in range(episodes):
		done = False
		obs = env.reset()

		episode_reward = 0

		while not done:
			action, _states = model.predict(obs)
			obs, reward, done, info = env.step(action)
			env.render()

			episode_reward += reward
		
		print(f"Episode: {episode} | Return: {episode_reward}")
		
		returns+=episode_reward
	
	print(f"AvgReward: {returns/episodes}")
	


if __name__ == '__main__':
	main()