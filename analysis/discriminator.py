import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        
        # 30x30x1
        self.conv1 = nn.Conv2d( 1,   32, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 30x30x32
        self.ln1   = nn.LayerNorm([32, 30, 30])
        self.conv2 = nn.Conv2d( 32,  64, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 30x30x64
        self.ln2   = nn.LayerNorm([64, 30, 30])        
        self.pool1 = nn.MaxPool2d(2, 2)                                                  # 15x15x64

        self.conv3 = nn.Conv2d( 64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 15x15x128
        self.ln3   = nn.LayerNorm([128, 15, 15])
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # 15x15x256
        self.ln4   = nn.LayerNorm([256, 15, 15])
        self.pool2 = nn.MaxPool2d(3, 3)                                                  # 15x15x256
        
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
        X = F.leaky_relu(self.ln1(self.conv1(X)))
        X = F.leaky_relu(self.ln2(self.conv2(X)))
        X = self.pool1(X)

        X = F.leaky_relu(self.ln3(self.conv3(X)))
        X = F.leaky_relu(self.ln4(self.conv4(X)))
        X = self.pool2(X)
        
        X = X.view(len(X), -1)
        MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.08164814, -0.02489864, 20.8446184, 
                                             -0.01204223,  0.02772552])
        STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.4557047,   5.38253167, 24.26102735, 
                                              2.69435522,  2.65776869])

        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        X = torch.cat([X, mom_point], dim=1)
        
        X = F.leaky_relu(self.fcbn1(self.fc1(X)))
        X = F.leaky_relu(self.fcbn2(self.fc2(X)))
        X = F.leaky_relu(self.fcbn3(self.fc3(X)))
        if TASK == 'WASSERSTEIN':
            return self.fc4(X)
        else:
            return torch.sigmoid(self.fc4(X))