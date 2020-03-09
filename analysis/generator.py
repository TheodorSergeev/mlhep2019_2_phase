import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

NOISE_DIM = 10

# PositiveEnergyReLUGenerator
class ModelGConvTranspose(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelGConvTranspose, self).__init__()
        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, 20736)

        self.conv1 = spectral_norm(nn.ConvTranspose2d(256, 256, 3, stride=2, output_padding=1))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = spectral_norm(nn.ConvTranspose2d(256, 128, 3))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = spectral_norm(nn.ConvTranspose2d(128, 64, 3))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = spectral_norm(nn.ConvTranspose2d(64, 32, 3))
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = spectral_norm(nn.ConvTranspose2d(32, 16, 3))
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = spectral_norm(nn.ConvTranspose2d(16, 1, 3))
        
    def forward(self, z, ParticleMomentum_ParticlePoint):
        # for both electrons and photons
        MEAN_TRAIN_MOM_POINT = torch.Tensor([-0.02729166, -0.02461355, 20.90919671, 
                                             -0.00788923,  0.00720004])
        STD_TRAIN_MOM_POINT  = torch.Tensor([ 5.48167024,  5.43016916, 24.32682144,  
                                              2.69976438,  2.67467291])

        mom_point = (ParticleMomentum_ParticlePoint - MEAN_TRAIN_MOM_POINT) / STD_TRAIN_MOM_POINT
        #mom_point = ParticleMomentum_ParticlePoint
        x = torch.cat([z, mom_point], dim=1)
        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        EnergyDeposit = x.view(-1, 256, 9, 9)
        
        EnergyDeposit = F.leaky_relu(self.bn1(self.conv1(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn2(self.conv2(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn3(self.conv3(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn4(self.conv4(EnergyDeposit)))
        EnergyDeposit = F.leaky_relu(self.bn5(self.conv5(EnergyDeposit)))
        EnergyDeposit = F.relu(self.conv6(EnergyDeposit))

        return EnergyDeposit

class EnsembleGenerator(nn.Module):
    def __init__(self, modelA, modelB):
        super(EnsembleGenerator, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, noise, batch, pdg_arr):
        x1 = self.modelA(noise, batch)
        x2 = self.modelB(noise, batch)
        output = torch.zeros_like(x1)

        for i in range(len(pdg_arr)):
            if pdg_arr[i] == 11.:
                output[i] = x1[i]
            else: # (pdg_arr[i] == 22.)
                output[i] = x2[i]

        return output
