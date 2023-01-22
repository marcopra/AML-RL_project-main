
import torch as th
import torchvision
import torch.nn as nn
from gym import spaces
from torchvision.models import alexnet
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MyAlexNet(BaseFeaturesExtractor):
    """
    Neural network model used ad Feature Extractor in SAC
    """
    def __init__(self, 
    observation_space: spaces.Box, 
    features_dim: int = 256
    ):

        super().__init__(observation_space, features_dim)

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(5),  # 84x84 --> 94x94
            torchvision.transforms.RandomCrop(size=(84,84)) )   #Then crops it to 84x84 again

        n_input_channels = observation_space.shape[0]
        self.net = alexnet()
        self.net.features[0] = nn.Conv2d(in_channels=n_input_channels, out_channels=64, kernel_size=11, stride=4)
       
        self.net.classifier[6] = nn.Linear(in_features=4096, out_features=features_dim)
        self.net.classifier[5] = nn.Tanh()
        self.net.classifier[2] = nn.Tanh()
        
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        
       
        x = observations
        # Data Augmentation (DA)
        # x = self.aug_trans(x)
        x = self.net(x)
        return x

policy_kwargs = dict(
    features_extractor_class=MyAlexNet,
    features_extractor_kwargs=dict(features_dim=128) )