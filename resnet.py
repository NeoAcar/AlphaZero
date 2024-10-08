import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(19, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.backBone = nn.Sequential(
            *[ResBlock() for _ in range(19)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 73, kernel_size=1, padding=0, bias=True),
            nn.Flatten(),

        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        x = self.backBone(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        
        return value, policy
    
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x, inplace=True)
        return x
