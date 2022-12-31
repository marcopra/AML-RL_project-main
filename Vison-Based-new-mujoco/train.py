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
print("Using torch version: {}".format(torch.__version__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cuda', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100000, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()

def main():

    BUFFER_SIZE=1000000
    BATCH_SIZE=128
    GAMMA=0.99
    TAU=0.001       #Target Network HyperParameters Update rate
    LRA=0.0001      #LEARNING RATE ACTOR
    LRC=0.001       #LEARNING RATE CRITIC
    H1=400   #neurons of 1st layers TODO: Deprecated
    H2=300   #neurons of 2nd layers TODO: Deprecated

    MAX_EPISODES=30000 #number of episodes of the training
    MAX_STEPS=200    #max steps to finish an episode. An episode breaks early if some break conditions are met (like too much
                    #amplitude of the joints angles or if a failure occurs). In the case of pendulum there is no break 
                    #condition, hence no environment reset,  so we just put 1 step per episode. 
    buffer_start = 200 #initial warmup without training
    epsilon = 1
    epsilon_decay = 1./100000 #this is ok for a simple task like inverted pendulum, but maybe this would be set to zero for more
                        #complex tasks like Hopper; epsilon is a decay for the exploration and noise applied to the action is 
                        #weighted by this decay. In more complex tasks we need the exploration to not vanish so we set the decay
                        #to zero. TODO: set to 0

    PRINT_EVERY = 500 #Print info and plots about average reward every PRINT_EVERY
    BACKUP_EVERY = 999 #Backup of Critic, Target_Critic, Actor and Target_Actor every BACKUP_EVERY episodes
    POINT_DISTANCE = 50 # Distance of points for plot is of POINT_DISTANCE episodes 

    ALTERNATE_TRAINING = True # Choose if alternate the training of the Networks. If False simoultaneous training, if Trure alternate training 
    CRITIC_TRAINING = True # The training of the Actor and Criting is not simultaneous
    THRESHOLD = 0.04 # Threshold for loss for Critic Network
    N_EPISODES_IN_A_ROW = 500 # Minumum amount of episodes in a row in which the critic loss is below THESHOLD needed to stop the Critic Training 
    MINIMUM_STEPS = 25 

    CONTINUE_TRAINING = False # Continue training from Backup


    # Observation from Environment will be a dictionary containg the pixel obsarvation associated to the key `pixels`
    # The actual shape of env['pixels'] is (480, 480, 3)
    env = PixelObservationWrapper(make_env(domain="source", render_mode='rgb_array'))
    # env = make_env(domain="source", render_mode='rgb_array')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper


    # assert isinstance(env.observation_space, Box), "observation space must be continuous" Now the observation are not continuous since the environment retrieve images
    assert isinstance(env.action_space, Box), "action space must be continuous"

    # Resetting the environment to get the dictionary containing the image observation
    s, _ = env.reset()

    # # Visualization of image before processing
    # data = np.zeros((500, 500,3), dtype=np.uint8)
    # data[0::, 0::] = s['pixels']
    # img = Image.fromarray(data)
    # img.show()

    # Porcessing the frame
    s = obs_processing(s['pixels'])

    # # Visualization of image after processing
    # s = np.reshape(s, (84, 84))
    # data = np.zeros((84, 84), dtype=np.uint8)
    # data[0::, 0::] = s
    # img = Image.fromarray(data)
    # img.show()
    
    # Frame shape dimension (used to create the input for the nn)
    state_dim = s.shape
    
    # Action dimension (used to create the input for the nn)
    action_dim = env.action_space.shape[0]

    # Noise Ornestein Uhlenbeck Action TODO da rivedere su `continuous control with deep learning``
    noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    # Critic network creation -> Q-Network (estiamtion of the reward given (St, At))
    critic  = StateValue(state_dim, action_dim).to(args.device)

    # Actor network creation -> Policy network (choose an action givent St)
    actor = Policy(state_dim, action_dim).to(args.device)

    # Target Critic Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
    target_critic  = StateValue(state_dim, action_dim).to(args.device)

    # Target Actor Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
    target_actor = Policy(state_dim, action_dim).to(args.device)

    if CONTINUE_TRAINING is True:

        critic.load_state_dict(torch.load("actor_critic_backup/critic.pkl"), strict=True)
        target_critic.load_state_dict(torch.load("actor_critic_backup/target_critic.pkl"), strict=True)
        actor.load_state_dict(torch.load("actor_critic_backup/actor.pkl"), strict=True)
        target_actor.load_state_dict(torch.load("actor_critic_backup/target_actor.pkl"), strict=True)
        

    else:
        # Parameter for Target Critic Network copied from Critic Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(param.data)

        # Parameter for Target Actor Network copied from Actor Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(param.data)
    
    # Optimizer Selection
    q_optimizer  = opt.Adam(critic.parameters(),  lr=LRC)#, weight_decay=0.01)
    policy_optimizer = opt.Adam(actor.parameters(), lr=LRA)

    # MSE Loss Definition
    MSE = nn.MSELoss()

    # Memory Buffer Definition -> to make TODO: mettere arxIv di Playing Atari Games with deep rl
    memory = replayBuffer(BUFFER_SIZE)

    #set GPU for faster training
    cuda = torch.cuda.is_available() #check for CUDA
    device   = torch.device("cuda" if cuda else "cpu")

    print("Job will run on {}".format(device))

    # Plot lists
    plot_reward = []
    plot_policy = []
    plot_q = []
    plot_steps = []

    # Variables Reset
    best_reward = -np.inf
    saved_reward = -np.inf
    saved_ep = 0
    average_reward = 0
    global_step = 0
    #s = deepcopy(env.reset())
    critic_low_loss = 0

    # actor.train()
    # critic.train()

    # Training Loop
    for episode in range(MAX_EPISODES):
        
        with open('out.txt', 'a') as f:
            with redirect_stdout(f):
                if (episode+1) % 100 == 0:
                    print(episode+1)

        #print(episode)

        # Environment Reset
        s, _ = env.reset()

        # Domain Randomization For Training
        # env.set_random_parameters()

        # Copy the state
        s = deepcopy(s)

        # Reset the noise
        noise.reset()

        # Reset the variables
        ep_reward = 0.
        ep_q_value = 0.
        step=0

        q_loss = None
        # Single Episode Loop
        for step in range(MAX_STEPS):
            #loss=0

            global_step +=1
            
            # actor.eval()
            # Resizing and processing of observation
            s = obs_processing(s['pixels'])
            
            # Get the Action from the Actor given the state
            a = actor.get_action(s)

            # Training Mode for Actor Network (TODO DA RIVEDERE SE NECESSARIO)
            # actor.train()
 
            # Noise to add to actions (TODO: vedere risultati)
            # a += noise()*max(0, epsilon)

            # # Make sure that the action is valid
            # a = np.clip(a, -1., 1.)

            # Performing of the action
            s2, reward, terminal, info = env.step(a)
            # print(reward)

            # Memory addition of St,At,Rt (if performing the action), done, St+1 
            memory.add(s, a, reward, terminal, obs_processing(s2['pixels']))

            #keep adding experiences to the memory until there are at least minibatch size samples
            if memory.count() > buffer_start:

                # Sampling BATCH_SIZE (= 64 items) of St, At, Rt, terminal, St+1 
                s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(BATCH_SIZE)

                s_batch = torch.FloatTensor(s_batch).to(device)
                a_batch = torch.FloatTensor(a_batch).to(device)
                r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
                s2_batch = torch.FloatTensor(s2_batch).to(device)

                
                #compute loss for critic
                a2_batch = target_actor(s2_batch)
                target_q = target_critic(s2_batch, a2_batch) #detach to avoid updating target

                # The real reward + continuity with the next state
                y = r_batch + (1.0 - t_batch) * GAMMA * target_q.detach() # (1 - t_batch) is because if it is the terminal state there is not the added value
                # Predicted value
                q = critic(s_batch, a_batch)
                    
                # Trainong for the Critic Network
                if ALTERNATE_TRAINING is True:

                    if CRITIC_TRAINING is True:
                        # Q-Network (Critic): Loss computation and backward (training of Q-Network)
                        q_optimizer.zero_grad()

                        q_loss = MSE(q, y) #detach to avoid updating target
                        q_loss.backward()
                        q_optimizer.step()

                        #Soft update of the frozen target networks
                        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                            target_param.data.copy_(
                                target_param.data * (1.0 - TAU) + param.data * TAU
                            )
                    
                    else:

                        q_loss = MSE(q, y) #detach to avoid updating target
                        # Actor Network: Loss computation and backward (training of Actor Network)
                        policy_optimizer.zero_grad()
                        # The aim is to maximize the reward of the chosen action, so the loss of the Actor corresponds to
                        # the reward predicted by the Critic Network. The '-' is needed because the `.backward()` can perform
                        # only minimization problem. 
                        a2 = actor(s_batch)

                        policy_loss =  - critic(s_batch, a2)
                        ''' 2° Tentativo'''
                        # policy_loss =  - (r_batch + (1.0 - t_batch) * GAMMA * target_q)
                        policy_loss = policy_loss.mean()
                        policy_loss.backward()
                        policy_optimizer.step()
                    
                        #Soft update of the frozen target networks
                        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                            target_param.data.copy_(
                                target_param.data * (1.0 - TAU) + param.data * TAU
                            )

                else:
                    
                    q_loss = MSE(q, y) #detach to avoid updating target

                    # Q-Network (Critic): Loss computation and backward (training of Q-Network)
                    q_optimizer.zero_grad()
                    q_loss.backward()
                    q_optimizer.step()

                    policy_optimizer.zero_grad()
                    # The aim is to maximize the reward of the chosen action, so the loss of the Actor corresponds to
                    # the reward predicted by the Critic Network. The '-' is needed because the `.backward()` can perform
                    # only minimization problem. 
                    policy_loss =  - critic(s_batch, actor(s_batch))
                    ''' 2° Tentativo'''
                    # policy_loss =  - (r_batch + (1.0 - t_batch) * GAMMA * target_q)
                    policy_optimizer.zero_grad()
                    policy_loss = policy_loss.mean()
                    policy_loss.backward()
                    policy_optimizer.step()

                
                    #Soft update of the frozen target networks
                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - TAU) + param.data * TAU
                        )

                    #Soft update of the frozen target networks
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - TAU) + param.data * TAU
                        )


                    
                
            # St+1 becomes St
            s = deepcopy(s2)
            
            ep_reward += reward
            
            if terminal:
            #    noise.reset()
               break

        try:
            # Plot values
            if ((episode + 1)%POINT_DISTANCE) == 0:
                plot_reward.append([ep_reward, episode+1])
                plot_q.append([q_loss.cpu().data, episode+1])
                plot_steps.append([step+1, episode+1])

                if CRITIC_TRAINING is False or ALTERNATE_TRAINING is False:
                    plot_policy.append([policy_loss.cpu().data, episode+1])
                else:
                    plot_policy.append([0, episode+1])
        except:
            pass

        if ALTERNATE_TRAINING is True:
            # If the q_loss is below a certain threshold, increment the counter of the time the q_loss is low
            if q_loss is not None and (q_loss.cpu().data <= THRESHOLD):
                critic_low_loss += 1

            else:
                critic_low_loss = 0
            
            # If the q_loss is below a certain threshold for a certain number of episodes in a row, 
            # we can stop the Critic Training and we can start with the Actor training 
            if CRITIC_TRAINING is True and critic_low_loss >= N_EPISODES_IN_A_ROW:
                with open('out.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(f"-----SWITCH at ep n°{episode}------")
                print(f"-----SWITCH at ep n°{episode}------")
                CRITIC_TRAINING = False
                plot_policy = []

        # Average Run reward -> run corresponds to `PRINT_EVERY` episodes
        average_reward += ep_reward

        
        # Saving the model with the best rewaed in episode
        if ep_reward > best_reward:
            if ALTERNATE_TRAINING is False or CRITIC_TRAINING is False:
                torch.save(actor.state_dict(), 'models_saved/best_model.pkl') #Save the actor model for future testing
                best_reward = ep_reward
                saved_reward = ep_reward
                saved_ep = episode+1
                print("Last best model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))

        # Plot Section
        if (episode % PRINT_EVERY) == (PRINT_EVERY-1):    # print every print_every episodes
            torch.save(actor.state_dict(), f'models_saved/model{episode + 1}.pkl') #Save the actor model for future testing
            
            # Testing
            for episode in range(10):
                    episode_reward = 0
                    done = False
                    state, _ = env.reset()

                    while not done:
                        action = actor.get_action(obs_processing(state['pixels']))
                        state, reward, done, _  = env.step(action=action)
        

                        episode_reward += reward

                    with open('out.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(f"Episode: {episode} | Return: {episode_reward}")
            
            # Plot
            subplot(plot_reward, plot_policy, plot_q, plot_steps)

            with open('out.txt', 'a') as f:
                with redirect_stdout(f):
                    print('[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'.format(PRINT_EVERY) %
                        (episode + 1, global_step, average_reward / PRINT_EVERY))
                    print("Last best model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
            average_reward = 0 #reset average reward
    
        # Plot Section
        if (episode % 50) == (50-1):    # print every print_every episodes
    
            # Testing
            for episode in range(10):
                    episode_reward = 0
                    done = False
                    state, _ = env.reset()

                    while not done:
                        action = actor.get_action(obs_processing(state['pixels']))
                        state, reward, done, _  = env.step(action=action)
        

                        episode_reward += reward

                    with open('out.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(f"Episode: {episode} | Return: {episode_reward}")
            
    
        if (episode % BACKUP_EVERY) == 0:
            print("-------Backup Saved-------")
            torch.save(actor.state_dict(), f'actor_critic_backup/actor.pkl')
            torch.save(target_actor.state_dict(), f'actor_critic_backup/target_actor.pkl')
            torch.save(critic.state_dict(), f'actor_critic_backup/critic.pkl')
            torch.save(target_critic.state_dict(), f'actor_critic_backup/target_critic.pkl')


if __name__ == '__main__':
	main()