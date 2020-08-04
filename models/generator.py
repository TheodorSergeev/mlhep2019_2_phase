import torch
import torch.nn as nn
import torch.nn.functional as F


NOISE_DIM = 256

class Generator(nn.Module):
    def __init__(self, z_dim, act_func = F.relu):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.activation = act_func

        # 128 + 5 -> (+reshape) 128 x 2 x 2
        self.fc1 = nn.Linear(self.z_dim + 5, self.z_dim * 2 * 2)

        # Z x 2 x 2
        self.conv1 = nn.ConvTranspose2d(self.z_dim, 128, 3, stride=2, padding=1, output_padding=1) 
        # 128 x 8 x 8
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        # 64 x 16 x 16
        self.conv3 = nn.ConvTranspose2d(64, 32,  3, stride=2, padding=1, output_padding=1)
        # 32 x 32 x 32
        self.conv4 = nn.ConvTranspose2d(32, 1,  3, stride=2, padding=1, output_padding=1)
        # 1 x 32 x 32
        # crop
        # 1 x 30 x 30

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

                
    def forward(self, z, ParticleMomentum_ParticlePoint):
        x = torch.cat([z, ParticleMomentum_ParticlePoint], dim=1)
        #print(x.shape)
        x = F.leaky_relu(self.fc1(x))

        #print(x.shape)
        EnergyDeposit = x.view(-1, self.z_dim, 2, 2)

        #print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.bn1(self.conv1(EnergyDeposit)))
        #print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.bn2(self.conv2(EnergyDeposit)))
        #print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.bn3(self.conv3(EnergyDeposit)))
        #print(EnergyDeposit.shape)
        EnergyDeposit = self.activation(self.conv4(EnergyDeposit))
        #print(EnergyDeposit.shape)
        EnergyDeposit = EnergyDeposit[:,:,1:31,1:31]
        #print(EnergyDeposit.shape)
                
        return EnergyDeposit
