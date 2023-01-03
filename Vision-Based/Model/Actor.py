import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import fanin_


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, h = 64, init_w = 3e-3):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = h
        self.tanh = torch.nn.Tanh()

        self.conv = nn.Sequential(
            nn.Conv2d(state_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        
        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc[0].weight.data = fanin_(self.fc[0].weight.data.size())
        self.fc[1].weight.data .uniform_(-init_w, init_w)

    def forward(self, x):

        # TODO: Provare ReLu
        """
            Actor
        """
        # print(x.shape)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

    def get_action(self, state):
        #(n_samples, channels, height, width)
        state  = torch.FloatTensor(np.float32(state)).unsqueeze(0).cuda()
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

# # FIRST TRY
# class Policy(torch.nn.Module):
#     def __init__(self, state_space, action_space, h = 64, init_w = 3e-3):
#         super().__init__()
#         self.state_space = state_space
#         self.action_space = action_space
#         self.hidden = h
#         self.tanh = torch.nn.Tanh()

#         self.conv = nn.Sequential(
#             nn.Conv2d(state_space[0], 32, kernel_size=1, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=1, stride=1),
#             nn.ReLU()
#         )
#         """
#             Actor network
#         """
#         ########### 704
#         self.fc1_actor = torch.nn.Linear(3840, self.hidden)
#         self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
#         self.fc3_actor_action = torch.nn.Linear(self.hidden, action_space)
        
#         # self.init_weights(init_w)

#     def init_weights(self, init_w):
#         self.fc1_actor.weight.data = fanin_(self.fc1_actor.weight.data.size())
#         self.fc2_actor.weight.data = fanin_(self.fc2_actor.weight.data.size())
#         self.fc3_actor_action.weight.data.uniform_(-init_w, init_w)

#     def forward(self, x):

#         # TODO: Provare ReLu
#         """
#             Actor
#         """
#         x = self.conv(x)
#         # x = x.view(x.size(0), -1)
#         x = torch.flatten(x, 1)
#         x_actor = self.tanh(self.fc1_actor(x))
#         x_actor = self.tanh(self.fc2_actor(x_actor))
#         action = self.fc3_actor_action(x_actor)

#         return action

#     def get_action(self, state):
#         state  = torch.FloatTensor(np.float32(state)).unsqueeze(0).cuda()
#         action = self.forward(state)
#         return action.detach().cpu().numpy()[0]