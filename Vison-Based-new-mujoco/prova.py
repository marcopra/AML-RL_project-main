import torch
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from env_utils import *
from Model.Actor import Policy
from Model.Critic import StateValue
from utils import OrnsteinUhlenbeckActionNoise, replayBuffer, subplot, obs_processing
from contextlib import redirect_stdout

from torch import nn 
import torch.nn.functional as F 
import torch.optim as opt 
from tqdm import tqdm_notebook as tqdm
import random
from copy import copy, deepcopy
from collections import deque
import numpy as np

import torch
from torch import nn #needed for building neural networks
import torch.nn.functional as F #needed for activation functions
from env_utils import *
import torch.optim as opt #needed for optimisation
from tqdm import tqdm_notebook as tqdm
import random
from copy import copy, deepcopy
from collections import deque
import numpy as np
print("Using torch version: {}".format(torch.__version__))

BUFFER_SIZE=1000000
BATCH_SIZE=64
GAMMA=0.99
TAU=0.001       #Target Network HyperParameters Update rate
LRA=0.0001      #LEARNING RATE ACTOR
LRC=0.001       #LEARNING RATE CRITIC
H1=400   #neurons of 1st layers
H2=300   #neurons of 2nd layers

MAX_EPISODES=20000 #number of episodes of the training
MAX_STEPS=200    #max steps to finish an episode. An episode breaks early if some break conditions are met (like too much
                  #amplitude of the joints angles or if a failure occurs). In the case of pendulum there is no break 
                #condition, hence no environment reset,  so we just put 1 step per episode. 
buffer_start = 100 #initial warmup without training
epsilon = 1
epsilon_decay = 1./100000 #this is ok for a simple task like inverted pendulum, but maybe this would be set to zero for more
                     #complex tasks like Hopper; epsilon is a decay for the exploration and noise applied to the action is 
                     #weighted by this decay. In more complex tasks we need the exploration to not vanish so we set the decay
                     #to zero.
PRINT_EVERY = 500 #Print info about average reward every PRINT_EVERY

ENV_NAME = "CustomHopper-source-v0" # Put here the gym env name you want to play with
#check other environments to play with at https://gym.openai.com/envs/#mujoco


class replayBuffer(object):
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

#set GPU for faster training
cuda = torch.cuda.is_available() #check for CUDA
device   = torch.device("cuda" if cuda else "cpu")

print("Job will run on {}".format(device))

def fanin_(size):
    fan_in = size[0]
    weight = 1./np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-weight, weight)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=3e-3):
        super(Critic, self).__init__()
                
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        #self.bn1 = nn.BatchNorm1d(h1)
        
        self.linear2 = nn.Linear(h1+action_dim, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
        self.linear3 = nn.Linear(h2, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.flatten(state, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(torch.cat([x,action],1))
        
        x = self.relu(x)
        x = self.linear3(x)
        
        return x
    

class Actor(nn.Module): 
    def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=0.003):
        super(Actor, self).__init__()
        
        #self.bn0 = nn.BatchNorm1d(state_dim)
        
        self.linear1 = nn.Linear(state_dim, h1)
        self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
        #self.bn1 = nn.BatchNorm1d(h1)
        
        self.linear2 = nn.Linear(h1, h2)
        self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
        
        #self.bn2 = nn.BatchNorm1d(h2)
        
        self.linear3 = nn.Linear(h2, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        #state = self.bn0(state)
        x = torch.flatten(state, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.tanh(x)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

class OrnsteinUhlenbeckActionNoise:
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
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


env = PixelObservationWrapper(make_env(domain="source", render_mode='rgb_array'))

s, _ = env.reset()
state_dim = obs_processing(s['pixels']).shape
action_dim = env.action_space.shape[0]


print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

# critic  = Critic(state_dim[2]*state_dim[1], action_dim).to(device)
# actor = Actor(state_dim[2]*state_dim[1], action_dim).to(device)

# target_critic  = Critic(state_dim[2]*state_dim[1], action_dim).to(device)
# target_actor = Actor(state_dim[1]*state_dim[1], action_dim).to(device)

critic  = StateValue(state_dim, action_dim).to(device)
actor = Policy(state_dim, action_dim).to(device)

target_critic  = StateValue(state_dim, action_dim).to(device)
target_actor = Policy(state_dim, action_dim).to(device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)
    
q_optimizer  = opt.Adam(critic.parameters(),  lr=LRC)#, weight_decay=0.01)
policy_optimizer = opt.Adam(actor.parameters(), lr=LRA)

MSE = nn.MSELoss()

memory = replayBuffer(BUFFER_SIZE)

plot_reward = []
plot_policy = []
plot_q = []
plot_steps = []

best_reward = -np.inf
saved_reward = -np.inf
saved_ep = 0
average_reward = 0
global_step = 0
#s = deepcopy(env.reset())

for episode in range(MAX_EPISODES):

    with open('out.txt', 'a') as f:
        with redirect_stdout(f):
            print(episode)

    s, _ = env.reset()
    s = deepcopy(s)
    #noise.reset()

    ep_reward = 0.
    ep_q_value = 0.
    step=0

    for step in range(MAX_STEPS):

        #loss=0
        global_step +=1
        # epsilon -= epsilon_decay
        #actor.eval()
        s = deepcopy(obs_processing(s['pixels']))
        a = actor.get_action(s)
        #actor.train()

        # a += noise()*max(0, epsilon)
        # a = np.clip(a, -1., 1.)

        s2, reward, terminal, info = env.step(a)
        # if step + 1 % 10 == 0:
        #     print("reward2: ", reward)


        memory.add(s, a, reward, terminal, obs_processing(s2['pixels']))

        #keep adding experiences to the memory until there are at least minibatch size samples
        
        if memory.count() > buffer_start:
            s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(BATCH_SIZE)
            
            s_batch = torch.FloatTensor(s_batch).to(device)
            a_batch = torch.FloatTensor(a_batch).to(device)
            r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
            t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
            s2_batch = torch.FloatTensor(s2_batch).to(device)

            # print("reward3: ", r_batch)
            
            #compute loss for critic
            a2_batch = target_actor(s2_batch)
            target_q = target_critic(s2_batch, a2_batch) #detach to avoid updating target
            y = r_batch + (1.0 - t_batch) * GAMMA * target_q.detach()
            q = critic(s_batch, a_batch)
            
            q_optimizer.zero_grad()
            q_loss = MSE(q, y) #detach to avoid updating target
            # print(q_loss)
            q_loss.backward()
            q_optimizer.step()
            
            #compute loss for actor
            policy_optimizer.zero_grad()
            policy_loss = - critic(s_batch, actor(s_batch))
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            policy_optimizer.step()
            
            #soft update of the frozen target networks
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU
                )

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - TAU) + param.data * TAU
                )

        s = deepcopy(s2)
        ep_reward += reward


        #if terminal:
        #    noise.reset()
        #    break

    try:
        if (episode + 1)%50 == 0:
            plot_reward.append([ep_reward, episode+1])
            plot_policy.append([policy_loss.cpu().data, episode+1])
            plot_q.append([q_loss.cpu().data, episode+1])
            plot_steps.append([step+1, episode+1])
    except:
        continue
    average_reward += ep_reward
    
    if ep_reward > best_reward:
        torch.save(actor.state_dict(), 'models_saved/best_model.pkl') #Save the actor model for future testing
        best_reward = ep_reward
        saved_reward = ep_reward
        saved_ep = episode+1
        print("Last model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))

    
    if (episode % PRINT_EVERY) == (PRINT_EVERY-1):    # print every print_every episodes
        torch.save(actor.state_dict(), f'models_saved/model{episode + 1}.pkl') #Save the actor model for future testing
        for episode in range(10):
                episode_reward = 0
                done = False
                state, _ = env.reset()

                while not done:
                    action = actor.get_action(obs_processing(state['pixels']))
                    state, reward, done, _  = env.step(action=action)
    

                    episode_reward += reward
                from contextlib import redirect_stdout

                with open('out.txt', 'a') as f:
                    with redirect_stdout(f):
                        print(f"Episode: {episode} | Return: {episode_reward}")
        subplot(plot_reward, plot_policy, plot_q, plot_steps)
        # r = list(zip(*plot_reward))
        # plt.plot(list(r[1]), list(r[0]), 'r') #row=0, col=0
        # plt.show()
        with open('out.txt', 'a') as f:
            with redirect_stdout(f):
                print('[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'.format(PRINT_EVERY) %
                    (episode + 1, global_step, average_reward / PRINT_EVERY))
                print("Last model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
        average_reward = 0 #reset average reward

    if (episode + 1) % 1000 == 0:
        torch.save(actor.state_dict(), f'actor_critic_backup/actor.pkl')
        torch.save(critic.state_dict(), f'actor_critic_backup/critic.pkl')