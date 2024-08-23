import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
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
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2)  # Output: 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)           # Output: 6x6
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)           # Output: 4x4
        
        # Calculate the output size of the conv layers
        conv_out_size = self._get_conv_out((1, 32, 32))
        print(conv_out_size)
        self.fc4 = nn.Linear(conv_out_size, 64)
        self.head = nn.Linear(64, n_actions)

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