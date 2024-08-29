import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import math
class DQNbn(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQNbn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.feature_size = self._get_conv_out((3, 210, 160))
        self.fc4 = nn.Linear(self.feature_size, 512)
        self.head = nn.Linear(512, n_actions)
    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)

class DQN(nn.Module):
    def __init__(self, in_channels=1, n_actions=4):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        # Adjusted convolutional layers for a 32x32 input
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)  # Output: 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)           # Output: 6x6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)           # Output: 4x4
        
        # Calculate the output size of the conv layers
        conv_out_size = self._get_conv_out((1, 16, 16))
        print(conv_out_size)
        self.fc4 = nn.Linear(conv_out_size, 256)
        self.head = nn.Linear(256, n_actions)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        # x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = F.relu(self.fc4(x))
        return self.head(x)
    
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class NoisyDQN(nn.Module):
    def __init__(self, in_channels=1, n_actions=4):
        super(NoisyDQN, self).__init__()
        
        # Define convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2),  # Output: 7x7
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),           # Output: 3x3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1)          # Output: 1x1
        )
        
        # Calculate the output size after convolutional layers
        self._calculate_conv_output_shape(in_channels)
        
        # Define noisy linear layers with reduced input size
        self.noisy1 = NoisyLinear(160, 128)
        self.noisy2 = NoisyLinear(128, n_actions)
    
    def _calculate_conv_output_shape(self, in_channels):
        dummy_input = torch.zeros(1, in_channels, 16, 16)
        x = dummy_input.float() / 255
        x = self.features(x)
        self.conv_output_size = x.view(1, -1).size(1)  # Flatten and get the size
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.float() / 255
        x = self.features(x)
        print(x.shape)
        x = x.view(batch_size, -1)  # Flatten the tensor
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()