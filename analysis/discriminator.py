class Baseline_ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3)
        
        # size
        self.fc1 = nn.Linear(2592 + 5, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):
        EnergyDeposit = self.dropout(F.leaky_relu(self.bn1(self.conv1(EnergyDeposit))))
        EnergyDeposit = self.dropout(F.leaky_relu(self.bn2(self.conv2(EnergyDeposit))))
        EnergyDeposit = F.leaky_relu(self.conv3(EnergyDeposit))
        EnergyDeposit = F.leaky_relu(self.conv4(EnergyDeposit)) # 32, 9, 9
        EnergyDeposit = EnergyDeposit.view(len(EnergyDeposit), -1)
        
        t = torch.cat([EnergyDeposit, ParticleMomentum_ParticlePoint], dim=1)
        
        t = F.leaky_relu(self.fc1(t))
        t = F.leaky_relu(self.fc2(t))
        t = F.leaky_relu(self.fc3(t))
        if TASK == 'WASSERSTEIN':
            return self.fc4(t)
        else:
            return torch.sigmoid(self.fc4(t))


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3)
        
        # size
        self.fc1 = nn.Linear(2592 + 5, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, EnergyDeposit, ParticleMomentum_ParticlePoint):
        EnergyDeposit = self.dropout(F.leaky_relu(self.bn1(self.conv1(EnergyDeposit))))
        EnergyDeposit = self.dropout(F.leaky_relu(self.bn2(self.conv2(EnergyDeposit))))
        EnergyDeposit = F.leaky_relu(self.conv3(EnergyDeposit))
        EnergyDeposit = F.leaky_relu(self.conv4(EnergyDeposit)) # 32, 9, 9
        EnergyDeposit = EnergyDeposit.view(len(EnergyDeposit), -1)
        
        t = torch.cat([EnergyDeposit, ParticleMomentum_ParticlePoint], dim=1)
        
        t = F.leaky_relu(self.fc1(t))
        t = F.leaky_relu(self.fc2(t))
        t = F.leaky_relu(self.fc3(t))
        if TASK == 'WASSERSTEIN':
            return self.fc4(t)
        else:
            return torch.sigmoid(self.fc4(t))