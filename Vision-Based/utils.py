import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from copy import copy, deepcopy
from collections import deque

from matplotlib import pyplot as plt
from IPython.display import clear_output
import cv2

cv2.ocl.setUseOpenCL(False) # disable GPU usage by OpenCV

# Xavier inizialization for fan-in #TODO rivedere bene
def fanin_(size):
    """
    Fan-in Xavier Inizialization

    Paraemeters
    -----------
    `Size`

    Returns
    ---------
    `Weight`: Tensor

    """
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

def subplot(R, P, Q, S, show = False):
    '''
    Create a Plot with 4 subplot representing:
    - Reward
    - Policy Loss
    - Q loss
    - Max steps (in each episode)

    Parameters
    -----------
    R: list
        Rewards list

    P: list
        Policy loss list

    Q: list
        Q-loss list
    
    S: list
        Steps for each episode list

    show: bool
        Flag to show the plot
    '''
    r = list(zip(*R))
    p = list(zip(*P))
    q = list(zip(*Q))
    s = list(zip(*S))
    clear_output(wait=True)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,15))

    ax[0, 0].plot(list(r[1]), list(r[0]), 'r') #row=0, col=0
    ax[1, 0].plot(list(p[1]), list(p[0]), 'b') #row=1, col=0
    ax[0, 1].plot(list(q[1]), list(q[0]), 'g') #row=0, col=1
    ax[1, 1].plot(list(s[1]), list(s[0]), 'k') #row=1, col=1
    ax[0, 0].title.set_text('Reward')
    ax[1, 0].title.set_text('Policy loss')
    ax[0, 1].title.set_text('Q loss')
    ax[1, 1].title.set_text('Max steps')
    if show:
        plt.show()

    # TODO Test the saving update
    fig.savefig(f"plot/subplot_ep_{r[1][-1]}.jpg")

def obs_processing(frame, width = 84, height = 84):
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
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]


class OrnsteinUhlenbeckActionNoise:
    '''
    Class used to create the noise to add to an element TODO: aggiungere seito wikipedia

    Methods
    -------
    Reset()
        Reset the computations for noise
    '''
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        '''
        Reset the noise
        '''
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class replayBuffer(object):
    """
    The class represent the memory Buffer
    """
    def __init__(self, buffer_size, name_buffer=''):
        self.buffer_size=buffer_size  #choose buffer size
        self.num_exp=0
        self.buffer=deque()

    def add(self, s, a, r, t, s2):
        experience=(s, a, r, t, s2)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)

        s, a, r, t, s2 = map(np.stack, zip(*batch))

        return s, a, r, t, s2

    def clear(self):
        self.buffer = deque()
        self.num_exp=0