import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        # todo: read about dilation to replace max pooling
        # todo: try adding dropout

        # 30x30x1
        self.conv1 = nn.Conv2d( 1,   32, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 30x30x32
        self.bn1   = nn.BatchNorm2d (32)
        self.conv2 = nn.Conv2d( 32,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 30x30x64
        self.bn2   = nn.BatchNorm2d (64)
        self.pool1 = nn.MaxPool2d(2, 2)                                                  # 15x15x64
        
        self.conv3 = nn.Conv2d( 64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 15x15x128
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 15x15x256
        self.bn4   = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(3, 3)                                                  # 5x5x256
        
        # 5x5x256 = 6400
        self.fc1 = nn.Linear(6400 + 5, 1024) 
        self.fcbn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fcbn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 16) 
        self.fcbn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):

        X = EnergyDeposit
        X = self.conv1  (X)
        X = self.bn1    (X)
        X = F.leaky_relu(X)
        X = self.conv2  (X)
        X = self.bn2    (X)
        X = F.leaky_relu(X)
        X = self.pool1  (X)

        X = self.conv3  (X)
        X = self.bn3    (X)
        X = F.leaky_relu(X)
        X = self.conv4  (X)
        X = self.bn4    (X)
        X = F.leaky_relu(X)
        X = self.pool2  (X)
        
        X = X.view(len(X), -1)        
        X = torch.cat([X, ParticleMomentum_ParticlePoint], dim=1)
        
        X = F.leaky_relu(self.fc1(X))#self.fcbn1(self.fc1(X)))
        X = F.leaky_relu(self.fc2(X))#self.fcbn2(self.fc2(X)))
        X = F.leaky_relu(self.fc3(X))#self.fcbn3(self.fc3(X)))
        if TASK == 'WASSERSTEIN':
            return self.fc4(X)
        else:
            return torch.sigmoid(self.fc4(X))