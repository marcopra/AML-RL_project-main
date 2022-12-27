"""Sample script for training a control policy on the Hopper environment

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between TRPO, PPO, and SAC.
"""
from env_utils import *
from stable_baselines3 import PPO

def main():
    render_mode = None  # you do not want to render at training time

    env = make_env(domain="source", render_mode=render_mode)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    print("1. Body Masses: ", env.model.body_mass[1:])
    env.set_random_parameters()
    print("Body Masses: ", env.model.body_mass[1:])

    """
        TODO:

            - train a policy
            - test the policy
    """

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("my_model")
	

if __name__ == '__main__':
    main()