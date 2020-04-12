#!/usr/bin/python
import sys
import numpy as np
from analysis.generator import DeepGenerator, VAE, NOISE_DIM
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import os

def main():
    input_dir, output_dir = sys.argv[1:]
    
    data_val = np.load(input_dir + '/data_val.npz', allow_pickle=True)
    val_data_path_out = output_dir + '/data_val_prediction.npz'

    data_test = np.load(input_dir + '/data_test.npz', allow_pickle=True)
    test_data_path_out = output_dir + '/data_test_prediction.npz'
    
    dir_path = os.path.dirname(os.path.abspath(__file__))

    vae_pretrained = VAE()
    vae_pretrained.load_state_dict(torch.load(dir_path + '/vae8.pt', map_location=torch.device('cpu')))
    vae_pretrained.eval()

    generator_cpu = DeepGenerator(vae_pretrained)
    generator_cpu_dict = torch.load(dir_path + '/vae_gan.pt', map_location=torch.device('cpu'))
    generator_cpu.load_state_dict(generator_cpu_dict)
    generator_cpu.eval()
    
    # val
    ParticleMomentum_val = torch.tensor(data_val['ParticleMomentum']).float()
    ParticlePoint_val = torch.tensor(data_val['ParticlePoint'][:, :2]).float()
    ParticleMomentum_ParticlePoint_val = torch.cat([ParticleMomentum_val, ParticlePoint_val], dim=1)
    ParticlePDG_val = torch.tensor(data_val['ParticlePDG']).float()

    calo_dataset_val = utils.TensorDataset(ParticleMomentum_ParticlePoint_val, ParticlePDG_val)
    calo_dataloader_val = torch.utils.data.DataLoader(calo_dataset_val, batch_size=1024, shuffle=False)

    with torch.no_grad():
        EnergyDeposit_val = []
        for ParticleMomentum_ParticlePoint_val_batch, ParticlePDG_val_batch in tqdm(calo_dataloader_val):
            noise = torch.randn(len(ParticleMomentum_ParticlePoint_val_batch), NOISE_DIM)
            vae_energy_b = vae_pretrained.decode(noise, ParticleMomentum_ParticlePoint_val_batch)

            EnergyDeposit_val_batch = generator_cpu(noise, ParticleMomentum_ParticlePoint_val_batch, 
                                                           vae_energy_b).detach().numpy()
            EnergyDeposit_val.append(EnergyDeposit_val_batch)
        np.savez_compressed(val_data_path_out, 
                            EnergyDeposit=np.concatenate(EnergyDeposit_val, axis=0).reshape(-1, 30, 30))

        del EnergyDeposit_val
    del data_val; del ParticleMomentum_val; del ParticlePoint_val; del ParticleMomentum_ParticlePoint_val;
    del calo_dataset_val; calo_dataloader_val
    
    
    ParticleMomentum_test = torch.tensor(data_test['ParticleMomentum']).float()
    ParticlePoint_test = torch.tensor(data_test['ParticlePoint'][:, :2]).float()
    ParticleMomentum_ParticlePoint_test = torch.cat([ParticleMomentum_test, ParticlePoint_test], dim=1)
    ParticlePDG_test = torch.tensor(data_test['ParticlePDG']).float()    

    calo_dataset_test = utils.TensorDataset(ParticleMomentum_ParticlePoint_test, ParticlePDG_test)
    calo_dataloader_test = torch.utils.data.DataLoader(calo_dataset_test, batch_size=1024, shuffle=False)

    with torch.no_grad():
        EnergyDeposit_test = []
        for ParticleMomentum_ParticlePoint_test_batch, ParticlePDG_test_batch in tqdm(calo_dataloader_test):
            noise = torch.randn(len(ParticleMomentum_ParticlePoint_test_batch), NOISE_DIM)

            vae_energy_b = vae_pretrained.decode(noise, ParticleMomentum_ParticlePoint_test_batch)

            EnergyDeposit_test_batch = generator_cpu(noise, ParticleMomentum_ParticlePoint_test_batch, 
                                                           vae_energy_b).detach().numpy()

            EnergyDeposit_test.append(EnergyDeposit_test_batch)
        np.savez_compressed(test_data_path_out, 
                            EnergyDeposit=np.concatenate(EnergyDeposit_test, axis=0).reshape(-1, 30, 30))

        del EnergyDeposit_test
    del data_test; del ParticleMomentum_test; del ParticlePoint_test; del ParticleMomentum_ParticlePoint_test;
    del ParticlePDG_test;
    del calo_dataset_test; calo_dataloader_test

    return 0

if __name__ == "__main__":
    main()
