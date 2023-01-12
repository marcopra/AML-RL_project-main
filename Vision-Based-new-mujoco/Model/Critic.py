import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from utils import fanin_




class StateValue(nn.Module):
    def __init__(self, state_space, action_dim, h1=32, h2=32, init_w=3e-3):
        super(StateValue, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_space[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(4096 + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.init_weights(init_w)

    def init_weights(self, init_w):
        
        self.conv[0].weight.data = fanin_(self.conv[0].weight.data.size())
        self.conv[3].weight.data = fanin_(self.conv[3].weight.data.size())
        self.conv[6].weight.data = fanin_(self.conv[6].weight.data.size())
        self.fc[0].weight.data = fanin_(self.fc[0].weight.data.size())
        self.fc[2].weight.data = fanin_(self.fc[2].weight.data.size())
        self.fc[4].weight.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # print(state.shape)
        x = self.conv(state)
        x = torch.flatten(x, 1)
        x = torch.cat([x,action],1)
        x = self.fc(x)
        return x
    
    




# class StateValue(nn.Module):
#     def __init__(self, state_space, action_dim, h1=32, h2=32, init_w=3e-3):
#         super(StateValue, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(state_space[0], 32, kernel_size=1, stride=4), ##########
#             # nn.Conv2d(state_space, 32, kernel_size=1, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=1, stride=1),
#             nn.ReLU()
#         )
        
#         ############ 704
#         self.linear1 = nn.Linear(3840 + action_dim, h1)
#         self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
        
#         # #self.bn1 = nn.BatchNorm1d(h1)
        
#         self.linear2 = nn.Linear(h1, h2)
#         self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
                
#         self.linear3 = nn.Linear(h2, 1)
#         self.linear3.weight.data.uniform_(-init_w, init_w)

#         self.relu = nn.ReLU()
        
#     def forward(self, state, action):
        
#         x = self.conv(state)
#         # x = x.view(x.size(0), -1)
#         x = torch.flatten(x, 1)
#         x = self.relu(self.linear1(torch.cat([x,action],1)))
#         x = self.relu(self.linear2(x))
#         x = self.relu(self.linear2(x))
        
#         return x