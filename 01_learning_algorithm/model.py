import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, state_size):

        super(DQN, self).__init__()

        # state parameters
        # 10 column heights
        # 1 number of holes
        # 1 bumpiness
        # 1 total height
        # 1 complete lines

        self.fc1 = nn.Linear(state_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1) 

    def forward(self, x):

        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))

        return self.fc3(x)