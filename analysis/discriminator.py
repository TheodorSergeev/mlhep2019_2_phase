class Baseline_ModelD(nn.Module):
    def __init__(self):
        super(Baseline_ModelD, self).__init__()
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
        # todo: read about dilation to replace max pooling
        # todo: try out dropout

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
        self.pool2 = nn.MaxPool2d(5, 5)                                                  # 3x3x256
        
        # 3x3x256 = 2304
        self.fc1 = nn.Linear(2304 + 5, 128) 
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 1)
        
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
        
        X = F.leaky_relu(self.fc1(X))
        X = F.leaky_relu(self.fc2(X))
        if TASK == 'WASSERSTEIN':
            return self.fc3(X)
        else:
            return torch.sigmoid(self.fc3(X))