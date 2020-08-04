import numpy as np
import torch
import torch.utils.data as utils

def dataset_rotations_transform(data_arr):
    EnergyDeposit, ParticleMomentum, ParticlePoint, PDG = data_arr[:]
    N = len(PDG)

    transf_energy   = np.zeros((4, N, 1, 30, 30))
    transf_momentum = np.zeros((4, N, 3))
    transf_point    = np.zeros((4, N, 2))
    transf_pdg      = np.zeros((4, N, 1))

    for i in range(0,N):
        transf_energy  [0][i] = EnergyDeposit   [i]
        transf_momentum[0][i] = ParticleMomentum[i]
        transf_point   [0][i] = ParticlePoint   [i]
        transf_pdg     [0][i] = PDG             [i]

        for rotation_num in range(1,4):
            transf_energy  [rotation_num][i] = np.rot90(transf_energy[rotation_num - 1][i], axes=(1,2))
            
            transf_momentum[rotation_num][i][0] = -transf_momentum[rotation_num - 1][i][1] # p_x -> -p_y
            transf_momentum[rotation_num][i][1] = +transf_momentum[rotation_num - 1][i][0] # p_y -> p_x
            transf_momentum[rotation_num][i][2] = +transf_momentum[rotation_num - 1][i][2] # p_z -> p_z
            
            transf_point   [rotation_num][i][0] = -transf_momentum[rotation_num - 1][i][1] # x -> -y
            transf_point   [rotation_num][i][1] = +transf_momentum[rotation_num - 1][i][0] # y -> x

            transf_pdg[rotation_num][i] = PDG[i]

    transf_energy   = torch.tensor(np.concatenate(transf_energy,   axis=0)).float()
    transf_momentum = torch.tensor(np.concatenate(transf_momentum, axis=0)).float()
    transf_point    = torch.tensor(np.concatenate(transf_point,    axis=0)).float()
    transf_pdg      = torch.tensor(np.concatenate(transf_pdg,      axis=0)).float()
    return utils.TensorDataset(transf_energy, transf_momentum, transf_point, transf_pdg)


def get_datasets(data_arr, train_size, valid_size,
                 one_particle_transf=True, particle_id=11.,
                 rotate_transf=False, normalise_energies=False, logarithm_energies=False):
    np.random.seed(123)

    energy   = torch.tensor(data_arr['EnergyDeposit'].reshape(-1, 1, 30, 30)).float()
    momentum = torch.tensor(data_arr['ParticleMomentum']).float()
    point    = torch.tensor(data_arr['ParticlePoint'][:, :2]).float()
    pdg      = torch.tensor(data_arr['ParticlePDG']).float()
    data_arr = utils.TensorDataset(energy, momentum, point, pdg)

    if one_particle_transf:
        pdg = data_arr[:][3]
        one_particle_ind_arr = [i for i, x in enumerate(pdg) if x == particle_id]
        energy, momentum, point, pdg = data_arr[one_particle_ind_arr]
        data_arr = utils.TensorDataset(energy, momentum, point, pdg)

    if rotate_transf:
        data_arr = dataset_rotations_transform(data_arr)
    
    if normalise_energies:
        energy, momentum, point, pdg = data_arr[:]
        # standart scaling
        MEAN_ENERGY_MATRIX = energy.mean(axis=0).reshape(1,30,30)
        STD_ENERGY_MATRIX  = energy.std(axis=0).reshape(1,30,30)
        energy = (energy - MEAN_ENERGY_MATRIX) / STD_ENERGY_MATRIX

        # E \in [E_min, E_max] -> E \in [-1, 1]
        energy = energy / torch.max(energy.abs())
        #print('E_min = ', energy.min())
        #print('E_max = ', energy.max())
        data_arr = utils.TensorDataset(energy, momentum, point, pdg)

    if logarithm_energies:
        energy, momentum, point, pdg = data_arr[:]
        energy = torch.log(1.0 + energy)
        data_arr = utils.TensorDataset(energy, momentum, point, pdg)

    dataset_len = len(data_arr)
    ind_arr = np.random.permutation(dataset_len)
    train_ind_arr = ind_arr[0:train_size]
    valid_ind_arr = ind_arr[train_size:train_size+valid_size]

    energy, momentum, point, pdg = data_arr[train_ind_arr]
    train_dataset = utils.TensorDataset(energy, momentum, point, pdg)

    energy, momentum, point, pdg = data_arr[valid_ind_arr]
    valid_dataset = utils.TensorDataset(energy, momentum, point, pdg)

    # sanity check
    # print(len(np.unique(np.concatenate((train_ind_arr, valid_ind_arr)))))

    return train_dataset, valid_dataset, train_ind_arr, valid_ind_arr