"""Raw Script - idea: use a translator network which map image -> state"""

from env_utils import *
from alexnet import MyAlexNet
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from env_utils import *

from torch import nn 
import torch.nn.functional as F 
import torch.optim as opt 
from tqdm import tqdm_notebook as tqdm
import numpy as np
print("Using torch version: {}".format(torch.__version__))
cuda = torch.cuda.is_available() #check for CUDA
device  = torch.device("cuda" if cuda else "cpu")

def gray_and_resize(frame):
    '''
    This function retrieves a resized frame of shape (`width` x `heigt`), converted from _RGB_ to _GRAY Scale_

    Parameters
    -----------
    `frame`: ndarray
        Input src image
    
    `width`: int
        Value representing the width of the image

    `height`: int
        Value representing the height of the image

    Returns
    ----------
    Resized frame: ndarray
            Output resized image, with shape (`width` x `heigt`), converted from _RGB_ to _GRAY Scale_
    '''
    if type(frame) != np.ndarray:
        frame = frame['pixels']
    height = 84
    width= 84
        
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

def imageToPytorch(observation):
    return np.swapaxes(observation, 2, 0)


def main():

    domain = 'source'
    pixels_only = False

    env = PixelObservationWrapper(make_env(domain=domain, render_mode='rgb_array'), pixels_only= pixels_only)
    print(env.observation_space['state'].shape[0])
    print(env.observation_space['pixels'])

    print('State space:', env.observation_space)  # state-space    
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper

    translator = MyAlexNet(env.observation_space['pixels'], env.observation_space['state'].shape[0])
    # Optimizer Selection
    optimizer  = opt.Adam(translator.parameters(),  lr=0.001)
    n_episodes = 5000000

    path = os.getcwd()  +  "/translator_model/translator.pkl"

    for ep in range(n_episodes):  
        done = False
        state = env.reset()  # Reset environment to initial state
        real_state = []
        pred_state= []
        loss = 0

        while not done:  # Until the episode is over
            
            action = env.action_space.sample()  # Sample random action

            state, reward, done, info = env.step(action)  # Step the simulator to the next timestep
            real_state = torch.from_numpy(state['state'])
            
        
            frame = torch.from_numpy(imageToPytorch(gray_and_resize(state['pixels'])))
            prediction = translator(frame.unsqueeze(1).float())
            loss +=  nn.MSELoss()(real_state.unsqueeze(0).float(), prediction.float())
            pred_state.append(prediction[None, ...])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(ep, " - ",loss)
        torch.save(translator.state_dict(), "./translator_model/translator.pkl")
if __name__ == '__main__':
    main()