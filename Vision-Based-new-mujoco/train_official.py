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
import wandb
print("Using torch version: {}".format(torch.__version__))


def parse_args():
    parser = argparse.ArgumentParser()
    ##parser.add_argument('--debug', default=False, type=bool, help='Activate to work and plot locally')
    parser.add_argument('--debug', default=False, action='store_true', help='if active, the work is locall without the use of wandb')
    parser.add_argument('--backup_every', default=999, type=int, help='Backup every N episodes')
    parser.add_argument('--print_every', default=100, type=int, help='Print plots every N episodes')
    parser.add_argument('--point_distance', '-pd', default=10, type=int, help='Distance between points for plots')
    parser.add_argument('--from_checkpoint', '-ckpt', default=False, type=bool, help='Activate to start training from checkpoint')
    parser.add_argument('--stop_critic', default=None, type=int, help='When (Episode) to stop the Critic Network Training, if None -> training no stop')


    parser.add_argument('--buffer_size', default= 1000000, type=int, help='Max size of the Replay Buffer')
    parser.add_argument('--batch_size', '-bs', default= 32, type=int, help='Max size of the Replay Buffer')
    parser.add_argument('--buffer_start', default=100, type=int, help='Initial warmup of replay buffer without training')

    parser.add_argument('--max_episodes', default=80000, type=int, help='Max N of episodes')
    parser.add_argument('--max_steps', default=2000, type=int, help='Max N of steps in one episode')

    parser.add_argument('--noise', default=True, type=bool, help='Noise for training')
    parser.add_argument('--epsilon', default=1, type=int, help='Starting value for noise')
    parser.add_argument('--noise_decay', default= 1./10000, type=int, help='Noise decay')
    parser.add_argument('--noise_prob', default=0.5, type=int, help='Probability of adding noise during training')

    parser.add_argument('--backbone_actor', default='ddpg', type=str, help='Backbone for Actor [ddpg, td3]')
    parser.add_argument('--backbone_critic', default='ddpg', type=str, help='Backbone for Actor [ddpg, td3]')
    parser.add_argument('--domain_randomization', '-dr', default=False, type=bool, help='Add domain randomization during the training')

    # TODO: da mettere a target
    parser.add_argument('--domain_test', default="source", type=str, help='choose the domain for testing [source, target]')
    parser.add_argument('--stack_frames', default=1, type=int, help='N of stacked frames, must be >= 1')
    parser.add_argument('--scaled_frames', default=False, type=bool, help='Frames scaled between 0. and 1.')

    parser.add_argument('--gamma', default=0.99, type=int, help='Gamma value for Critic Network Loss, must be between 0. and 1.')
    parser.add_argument('--tau', default=0.001, type=int, help='Constant for soft update of networks, must be between 0. and 1.')
    parser.add_argument('--lra', default=0.0001, type=int, help='Learning Rate for Actor Network')
    parser.add_argument('--lrc', default=0.001, type=int, help='Learning Rate for Critic Network')


    return parser.parse_args()

args = parse_args()

def config():

    global DEBUG # TODO not implemented
    global BACKUP_EVERY #Backup of Critic, Target_Critic, Actor and Target_Actor every BACKUP_EVERY episodes
    global PRINT_EVERY  #Print info and plots about average reward every PRINT_EVERY
    global POINT_DISTANCE # Distance of points for plot is of POINT_DISTANCE episodes 
    global CONTINUE_TRAINING # Continue training from Backup
    global STOP_CRITIC  # When to stop the training of the Criting net
    global device

    global BUFFER_SIZE
    global BATCH_SIZE
    global buffer_start #initial warmup without training

    global MAX_EPISODES #number of episodes of the training
    global MAX_STEPS    #max steps to finish an episode. An episode breaks early if some break conditions are met (like too much
                        #amplitude of the joints angles or if a failure occurs). In the case of pendulum there is no break 
                        #condition, hence no environment reset,  so we just put 1 step per episode. 

    
    global epsilon
    global epsilon_decay #this is ok for a simple task like inverted pendulum, but maybe this would be set to zero for more
                                    #complex tasks like Hopper; epsilon is a decay for the exploration and noise applied to the action is 
                                    #weighted by this decay. In more complex tasks we need the exploration to not vanish so we set the decay
                                    #to zero. TODO: what if 0 or not
    global PERCENTAGE_FOR_NOISE # Exploitation-Exploration balancing 

    global BACKBONE_ACTOR #Backbone for actor Network TODO: Not Implemented
    global BACKBONE_CRITIC #Backbone for critic Network TODO: Not Implemented
    global DOMAIN_RANDOMIZATION 
    global DOMAIN_TEST_ENV

    global STACK_FRAMES
    global SCALED_FRAMES

    global GAMMA
    global TAU       #Target Network HyperParameters Update rate
    global LRA       #LEARNING RATE ACTOR
    global LRC       #LEARNING RATE CRITIC
    global NOISE

    #TODO inserire controlli

    DEBUG = args.debug 
    BACKUP_EVERY = args.backup_every 
    PRINT_EVERY = args.print_every 
    POINT_DISTANCE = args.point_distance 
    CONTINUE_TRAINING = args.from_checkpoint
    STOP_CRITIC = args.stop_critic
    #set GPU for faster training
    cuda = torch.cuda.is_available() #check for CUDA
    device  = torch.device("cuda" if cuda else "cpu")

    print("Job will run on {}".format(device))

    BUFFER_SIZE=args.buffer_size
    BATCH_SIZE=args.batch_size
    buffer_start = args.buffer_start
    MAX_EPISODES=args.max_episodes
    MAX_STEPS= args.max_steps

    NOISE = args.noise
    epsilon = args.epsilon
    epsilon_decay = args.noise_decay
    PERCENTAGE_FOR_NOISE = args.noise_prob
    
    BACKBONE_ACTOR = args.backbone_actor
    BACKBONE_CRITIC = args.backbone_critic
    DOMAIN_RANDOMIZATION = args.domain_randomization

    DOMAIN_TEST_ENV = args.domain_test
    STACK_FRAMES = args.stack_frames
    SCALED_FRAMES = args.scaled_frames

    GAMMA=args.gamma
    TAU=args.tau  
    LRA=args.lra    
    LRC=args.lrc  
    
    # Wandb initialization
    
    wandb.init(project="test-project", entity="aml-rl_project")
    
    

    wandb.config.update(args)
    
   
    
     
    
    # TODO: inseriregeneral path e creazione di directory
    # serve ./plot, ./actor_critic_backup, ./models_saved
    
    return

def main():

    config()

    global epsilon
    global NOISE
    
    env = my_make_env(stack_frames=STACK_FRAMES, scale=SCALED_FRAMES )
    test_env = my_make_env(stack_frames=STACK_FRAMES, scale=SCALED_FRAMES, domain=DOMAIN_TEST_ENV)

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper


    # assert isinstance(env.observation_space, Box), "observation space must be continuous" Now the observation are not continuous since the environment retrieve images
    assert isinstance(env.action_space, Box), "action space must be continuous"

    # Resetting the environment to get the dictionary containing the image observation
    s, _ = env.reset()

    # # Visualization of image after processing
    # DA PROVARE
    # data = np.zeros((84, 84), dtype=np.uint8)
    # try:
    #   s = np.reshape(s, (84, 84))
    # except:
    #   s = np.reshape(s[0], (84, 84))
    # data[0::, 0::] = s
    # img = Image.fromarray(data)
    # img.show()
    
    # Frame shape dimension (used to create the input for the nn)
    state_dim = s.shape
    
    # Action dimension (used to create the input for the nn)
    action_dim = env.action_space.shape[0]

    if NOISE is True:
        # Noise Ornestein Uhlenbeck Action TODO da rivedere su `continuous control with deep learning``
        noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    # Critic network creation -> Q-Network (estiamtion of the reward given (St, At))
    critic  = StateValue(state_dim, action_dim).to(device)

    # Actor network creation -> Policy network (choose an action givent St)
    actor = Policy(state_dim, action_dim).to(device)

    # Target Critic Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
    target_critic  = StateValue(state_dim, action_dim).to(device)

    # Target Actor Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
    target_actor = Policy(state_dim, action_dim).to(device)

    if CONTINUE_TRAINING is True:

        critic.load_state_dict(torch.load("actor_critic_backup/critic.pkl"), strict=True)
        target_critic.load_state_dict(torch.load("actor_critic_backup/target_critic.pkl"), strict=True)
        actor.load_state_dict(torch.load("actor_critic_backup/actor.pkl"), strict=True)
        target_actor.load_state_dict(torch.load("actor_critic_backup/target_actor.pkl"), strict=True)
        
    else:

        # Parameter for Second Critic Network copied from Critic Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(param.data)

        # Parameter for Target Actor Network copied from Actor Network -> TODO: mettere arxIv di Playing Atari Games with deep rl
        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(param.data)
    
    # Optimizer Selection
    q_optimizer  = opt.Adam(critic.parameters(),  lr=LRC)#, weight_decay=0.01)
    policy_optimizer = opt.Adam(actor.parameters(), lr=LRA)


    # Memory Buffer Definition -> to make TODO: mettere arxIv di Playing Atari Games with deep rl
    memory = replayBuffer(BUFFER_SIZE)

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
  
    # Training Loop
    for episode in range(MAX_EPISODES):
        
        with open('out.txt', 'a') as f:
            with redirect_stdout(f):
                if (episode+1) % 100 == 0:
                    print(episode+1)
                    

        # Environment Reset
        s, _ = env.reset()

        if DOMAIN_RANDOMIZATION:
            # Domain Randomization For Training
            env.set_random_parameters()

        # Copy the state
        s = deepcopy(s)

        if NOISE:
            # Reset the noise
            noise.reset()

        # Reset the variables
        ep_reward = 0.
        step=0

        # Single Episode Loop
        for step in range(MAX_STEPS):
            #loss=0

            global_step +=1
            
            # Get the Action from the Actor given the state
            a = actor.get_action(s)

            # Training Mode for Actor Network (TODO DA RIVEDERE SE NECESSARIO)
            # actor.train()
 
            if NOISE and random.random() > PERCENTAGE_FOR_NOISE:

                
                # Noise to add to actions (TODO: vedere risultati)
                a += noise()*max(0, epsilon)

                # # Make sure that the action is valid
                a = np.clip(a, -1., 1.)

                epsilon-=epsilon_decay
                
                if epsilon <=0:
                    NOISE = False

            # Performing of the action
            s2, reward, terminal, info = env.step(a)
            # print(reward)

            # Memory addition of St,At,Rt (if performing the action), done, St+1 
            memory.add(s, a, reward, terminal, s2)

            #keep adding experiences to the memory until there are at least minibatch size samples
            if memory.count() > buffer_start:

                # Sampling BATCH_SIZE (= 64 items) of St, At, Rt, terminal, St+1 
                s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(BATCH_SIZE)

                s_batch = torch.FloatTensor(s_batch).to(device)
                a_batch = torch.FloatTensor(a_batch).to(device)
                r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
                t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
                s2_batch = torch.FloatTensor(s2_batch).to(device)

                
               # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models
                actions_next = target_actor(s2_batch)
                Q_targets_next = target_critic(s2_batch, actions_next)
                # Compute Q targets for current states (y_i)
                Q_targets = r_batch + (GAMMA * Q_targets_next.detach() * (1 - t_batch))
                # Compute critic loss
                Q_expected = critic(s_batch, a_batch)
                q_loss = F.mse_loss(Q_expected, Q_targets)
                # Minimize the loss
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actions_pred = actor(s_batch)
                policy_loss = -critic(s_batch, actions_pred).mean()
                # Minimize the loss
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # ----------------------- update target networks ----------------------- #
            
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
        
                break
            
        try:
            # Plot values
            if ((episode + 1)%POINT_DISTANCE) == 0:
                plot_reward.append([ep_reward, episode+1])
                plot_q.append([q_loss.cpu().data, episode+1])
                plot_steps.append([step+1, episode+1])
                
                plot_policy.append([policy_loss.cpu().data, episode+1])
                
                
                
                
        except: 
         pass

        # Average Run reward -> run corresponds to `PRINT_EVERY` episodes
        average_reward += ep_reward
        
        # Saving the model with the best rewaed in episode
        if ep_reward > best_reward:
            try:
                torch.save(actor.state_dict(), 'models_saved/best_model.pkl') #Save the actor model for future testing
                best_reward = ep_reward
                saved_reward = ep_reward
                saved_ep = episode+1
                print("Last best model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
            except:
                print("Failed saving")

        # Plot Section
        if (episode % PRINT_EVERY) == (PRINT_EVERY-1):    # print every print_every episodes
            torch.save(actor.state_dict(), f'models_saved/model{episode + 1}.pkl') #Save the actor model for future testing
            
            
            # Testing
            for episode in range(10):
                    episode_reward = 0
                    done = False
                    state, _ = test_env.reset()

                    while not done:
                        action = actor.get_action(state)
                        state, reward, done, _  = test_env.step(action=action)
        

                        episode_reward += reward

                    with open('out.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(f"Episode: {episode} | Return: {episode_reward}")
            
            # Plot
            subplot(plot_reward, plot_policy, plot_q, plot_steps)
            
            ## wandb grafici 
            
            table1 = wandb.Table(data= plot_reward, columns = ["steps", "reward"])
            line_plot_pr = wandb.plot.line(table1, x='steps', y='reward', title='Plot Reward')
            wandb.log({'line_plot_pr': line_plot_pr})
            
            table2 = wandb.Table(data= plot_policy, columns = ["steps", "reward"])
            line_plot_pp = wandb.plot.line(table2, x='steps', y='reward', title='Plot Policy')
            wandb.log({'line_plot_pp': line_plot_pp})
            
            table3 = wandb.Table(data= plot_q, columns = ["steps", "reward"])
            line_plot_q = wandb.plot.line(table3, x='steps', y='reward', title='Plot Q')
            wandb.log({'line_plot_q': line_plot_q})
            
            table4 = wandb.Table(data= plot_steps, columns = ["steps", "reward"])
            line_plot_s = wandb.plot.line(table4, x='steps', y='reward', title='Plot Steps')
            wandb.log({'line_plot_steps': line_plot_s})
            
            
           
            with open('out.txt', 'a') as f:
                with redirect_stdout(f):
                    print('[%6d episode, %8d total steps] average reward for past {} iterations: %.3f'.format(PRINT_EVERY) %
                        (episode + 1, global_step, average_reward / PRINT_EVERY))
                    print("Last best model saved with reward: {:.2f}, at episode {}.".format(saved_reward, saved_ep))
            average_reward = 0 #reset average reward
    
        # print section
        if (episode % 100) == (100-1):    # print every print_every episodes
    
            # Testing
            for episode in range(10):
                    episode_reward = 0
                    done = False
                    state, _ = test_env.reset()

                    while not done:
                        action = actor.get_action(state)
                        state, reward, done, _  = test_env.step(action=action)
        

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