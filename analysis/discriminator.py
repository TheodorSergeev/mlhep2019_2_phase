import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        
        # 30x30x1
        self.conv1 = spectral_norm(nn.Conv2d( 1,   32, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 30x30x32
        self.conv2 = spectral_norm(nn.Conv2d( 32,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 30x30x64
        self.pool1 = nn.MaxPool2d(2, 2)                                                                 # 15x15x64

        self.conv3 = spectral_norm(nn.Conv2d( 64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 15x15x128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))) # 15x15x256
        self.pool2 = nn.MaxPool2d(3, 3)                                                                 # 15x15x256
        
        # 5x5x256 = 6400
        self.fc1 = spectral_norm(nn.Linear(6400 + 5, 128))
        self.fc2 = spectral_norm(nn.Linear(128,32))
        self.fc3 = spectral_norm(nn.Linear(32,1))
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):
        X = EnergyDeposit
        X = F.leaky_relu(self.conv1(X))
        X = F.leaky_relu(self.conv2(X))
        X = self.pool1(X)

        X = F.leaky_relu(self.conv3(X))
        X = F.leaky_relu(self.conv4(X))
        X = self.pool2(X)
        
        X = X.view(len(X), -1)

        # for both electrons and photons
        MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.02729166, -0.02461355, 20.90919671, 
                                             -0.00788923,  0.00720004])
        STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.48167024,  5.43016916, 24.32682144,  
                                              2.69976438,  2.67467291])
        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        X = torch.cat([X, mom_point], dim=1)
        
        X = F.leaky_relu(self.fc1(X))
        X = F.leaky_relu(self.fc2(X))
        #X = F.leaky_relu(self.fcbn3(self.fc3(X)))
        if TASK in ['WASSERSTEIN', 'HINGE']:
            return self.fc3(X)
        else:
            return torch.sigmoid(self.fc3(X))