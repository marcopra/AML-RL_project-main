'''
    Unused File, but it can be used as Reference
'''
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Agent(object):
    def __init__(self, policy, state_value, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.state_value = state_value.to(self.train_device)
        self.optimizerAction = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.optimizerCritic = torch.optim.Adam(state_value.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.discountedRewards = []
        self.q_value = []
        self.next_q_value = []

        

    def update_policy(self):
       
        # Inizialization
        lossCritic = 0
        lossActor = 0
        
        assert len(self.rewards) == len(self.q_value), "rewards and Q-Values arrays must have the same size"

        # Losses Computation
        for lp,q_t, q_t_1, r_t, done in zip(self.action_log_probs, self.q_value, self.next_q_value, self.rewards, self.done):
            # If terminal state...
            if done:
                lossCritic += ((r_t) - q_t)**2
                lossActor += (r_t  - q_t)*lp
            else:
                lossCritic += ((r_t + self.gamma*q_t_1.detach()) - q_t)**2
                lossActor += (r_t + self.gamma*q_t_1 - q_t)*lp
            
        # Averaged loss
        lossCritic = lossCritic/len(self.rewards)
        lossActor = -lossActor/len(self.rewards)

        #Pytorch keep tracks of the losses' history and it knows where to backpropagate (?????)
        losses = lossActor + lossCritic     

        self.optimizerAction.zero_grad()
        self.optimizerCritic.zero_grad()

        losses.backward()
        
        self.optimizerCritic.step()
        self.optimizerAction.step()

        # Reset histories and parameters
        self.reset_all()
        return        

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            q_value = self.state_value(x)

            return action, action_log_prob, q_value

    def store_outcome(self, state, next_state, action_log_prob, q_value, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.done.append(done)
        self.q_value.append(q_value)
        self.next_q_value.append(self.state_value(torch.from_numpy(next_state).float().to(self.train_device)))

        self.discountedRewards = []
        for t in range(len(self.rewards)):
            G = 0.0
            for k, r in enumerate(self.rewards[t:]):
                G += (self.gamma**k)*r
            
            self.discountedRewards.append(G)

    # Reset histories and parameters
    def reset_all(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.q_value = []
        self.discountedRewards = []
        self.next_q_value = []

