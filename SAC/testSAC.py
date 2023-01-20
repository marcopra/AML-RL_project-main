"""Test an RL agent on the OpenAI Gym Hopper environment"""

import numpy as np
from env.custom_hopper import *
from stable_baselines3 import SAC

def test_on(domain , model_path):

	src_model = model_path
	print("Model "     , src_model)
	filename = src_model.split('/')[-1].removesuffix('.zip')
	env_params = filename.split('_')

	env_args: dict = {}

	for param in env_params:
		key, val = param.split('-')
		if val == 'False':
			env_args[key] = False
		elif val == 'True':
			env_args[key] = True
		else:
			env_args[key] = val

	env = my_make_env(PixelObservation= bool(env_args['img']), 
							stack_frames= int(env_args['nf']), 
							scale= bool(env_args['scaled']),
							domain= domain)
		
	model = SAC.load(src_model, device='cpu')
	
	episodes = 50
	all_rewards = []
	
	for episode in range(episodes):
		done = False
		obs = env.reset()
		episode_reward = 0

		while not done:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, _info = env.step(action)
			episode_reward += reward
		
		print(f"Episode: {episode} | Return: {episode_reward}")

		all_rewards.append(episode_reward)


	print(f"AvgReward: {np.mean(all_rewards)} and devSTD: {np.std(all_rewards)}")
	

if __name__ == '__main__':

	domain = 'target'
	filepath = 'trains/resnet/dr-done/alg-sac_dom-source_img-True_ts-300000_nf-4_scaled-False.zip'
	test_on(domain=domain, model_path=filepath)