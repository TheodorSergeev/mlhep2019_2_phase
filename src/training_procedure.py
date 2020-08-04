from analysis.prd_score import *
from src.gan_losses     import *

from analysis.prd_score import compute_prd, compute_prd_from_embedding, _prd_to_f_beta
from sklearn.metrics    import auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm, tqdm_notebook
from IPython.display import clear_output

from src.plotting_functions import *


TASKS = ['KL', 'REVERSED_KL', 'WASSERSTEIN', 'HINGE']

TASK = 'HINGE'

# Additional things for Wasserstein GAN
LIPSITZ_WEIGHTS = False
clamp_lower, clamp_upper = -0.01, 0.01

GRAD_PENALTY = True                  # https://arxiv.org/abs/1704.00028
ZERO_CENTERED_GRAD_PENALTY = False   # https://arxiv.org/abs/1705.09367

# Small hack that can speed-up training and improve generalization
INSTANCE_NOISE = True                # https://arxiv.org/abs/1610.04490



def prd_auc(generated_example, real_example, reshape_size):
    precision, recall = compute_prd_from_embedding(
                            generated_example.reshape(reshape_size, -1), 
                            real_example.reshape(reshape_size, -1),
                            num_clusters=30, num_runs=100)
    return auc(precision, recall)


class TrainingProcedure:

    def __init__(self, loss_function_type, discriminator, generator,
                 train_dataloader, valid_dataloader,
                 batch_size, valid_size, device,
                 lr_dis = 4e-4, lr_gen = 1e-4,
                 dis_iter_num = 3, gen_iter_num = 1, noise_dim = 30,
                 loss = 'HINGE', INSTANCE_NOISE=INSTANCE_NOISE):

        self.current_epoch = 0
        self.loss_type = loss
        self.gan_loss = GANLosses(loss, device)
        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.BATCH_SIZE = batch_size
        self.VALID_SIZE = valid_size
        self.noise_dim  = noise_dim
        self.device = device

        self.INSTANCE_NOISE = INSTANCE_NOISE
        
        # array for saving the distributions that are generated on every step
        self.DRAW_ID = 8
        energy, _, _, _ = self.valid_dataloader.dataset[self.DRAW_ID]
        self.generated_examples_list = [energy] # first element is the real distribution

        self.best_models = [self.discriminator, self.generator]

        self.g_optimizer = optim.Adam(self.generator.parameters(), betas=(0.0, 0.999), lr=lr_gen)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), betas=(0.0, 0.999), lr=lr_dis)

        self.k_d = dis_iter_num
        self.k_g = gen_iter_num
        
        self.dis_epoch_loss,  self.gen_epoch_loss  = [], []
        self.predictions_dis, self.predictions_gen = [], []
        self.prd_auc = []  

        self.val_prd_auc = []

        return

    def run(self, epoch_num, use_separate_validation=False):
        for epoch in tqdm(np.arange(self.current_epoch, epoch_num)):      
            # Validation
            if use_separate_validation:
                self.validation()        
            
            # Training
            self.generator.train()
            self.discriminator.train()
            first = True

            for energy_b, mom_b, point_b, pdg_b in self.train_dataloader:
                
                energy_b, mom_b = energy_b.to(self.device), mom_b.to(self.device)
                point_b,  pdg_b = point_b .to(self.device), pdg_b.to(self.device)
                mom_point_b = torch.cat([mom_b.to(self.device), point_b.to(self.device)], dim=1)

                if first:
                    noise = torch.randn(len(energy_b), self.noise_dim).to(self.device)
                    energy_gen = self.generator(noise, mom_point_b)
                    
                    pred_dis_list = list(self.discriminator(energy_b,   mom_point_b).detach().cpu().numpy().ravel())
                    pred_gen_list = list(self.discriminator(energy_gen, mom_point_b).detach().cpu().numpy().ravel())
                    
                    self.predictions_dis.append(pred_dis_list)
                    self.predictions_gen.append(pred_gen_list)
                    
                # Optimize D
                dis_loss_item = self.train_epoch_dis(self.k_d, energy_b, mom_point_b)
                self.dis_epoch_loss.append(dis_loss_item)

                # Optimize G
                gen_loss_item = self.train_epoch_gen(self.k_g, energy_b, mom_point_b)
                self.gen_epoch_loss.append(gen_loss_item)

                if first:
                    reshape_size = pdg_b.shape[0] # = BATCH_SIZE or less
                    self.prd_auc.append(prd_auc(energy_gen.detach().cpu().numpy(),
                                                energy_b  .detach().cpu().numpy(), 
                                                reshape_size))
                    first = False
            
            # Save the new model if it is the best one (max training prd)
            if self.prd_auc[-1] == np.max(self.prd_auc):
                print("New best model")
                self.best_models = [self.discriminator, self.generator]

            # Plot training progress

            self.generator.eval()
            self.discriminator.eval()

            self.plot_and_draw(if_append_sample=True, use_separate_validation=use_separate_validation)
            self.current_epoch += 1  

        return

    def plot_and_draw(self, if_append_sample=False, use_separate_validation=False):
        with torch.no_grad():
            energy_b, mom_b, point_b, pdg_b = self.valid_dataloader.dataset[self.DRAW_ID]

            energy_b = energy_b.reshape((1,1,30,30))
            point_b  = point_b.reshape((1,2))
            mom_b    = mom_b.reshape((1,3))
            pdg_b    = pdg_b.reshape((1,1))

            noise = torch.randn(len(energy_b), self.noise_dim).to(self.device)
            mom_point_b = torch.cat([mom_b.to(self.device), point_b.to(self.device)], dim=1)
            energy_gen = self.generator(noise, mom_point_b)
            real_res = energy_b  .detach().cpu()[0]
            gen_res  = energy_gen.detach().cpu()[0]

            if if_append_sample:
                self.generated_examples_list.append(gen_res) # save the generated example

            if not use_separate_validation:
                self.val_prd_auc = self.prd_auc # заглушка, чтобы сильно не менять функцию построения графиков

            self.plot_training_progress(self.current_epoch, real_res, gen_res)
        return

    def train_epoch_dis(self, k_d, energy_b, mom_point_b):
        """
        Train discriminator for one epoch
        
            Parameters
            ----------
                k_d : int
                    Number of iterations to train for.
                
                energy_b : torch.Tensor
                    Batch of energy distributions.
                
                mom_point_b : torch.Tensor
                    Batch of concatenated particle momentum and entrance point.
                                
            Returns:
            ----------
                loss.item() : float
                    Loss on the last iteration of training.
        """

        for _ in range(k_d):
            noise = torch.randn(len(energy_b), self.noise_dim).to(self.device)
            energy_gen = self.generator(noise, mom_point_b)

            if self.INSTANCE_NOISE:
                energy_b   = add_instance_noise(energy_b,   self.device)
                energy_gen = add_instance_noise(energy_gen, self.device)
                
            loss = self.gan_loss.d_loss(self.discriminator(energy_gen, mom_point_b),
                                    self.discriminator(energy_b,   mom_point_b))
        
            coef = 0
            if GRAD_PENALTY:
                coef = +1.
            elif ZERO_CENTERED_GRAD_PENALTY:
                coef = -1.

            loss += coef * self.gan_loss.calc_gradient_penalty(self.discriminator,
                                                            energy_gen.data,
                                                            mom_point_b,
                                                            energy_b.data)
            self.d_optimizer.zero_grad()
            loss.backward()
            self.d_optimizer.step()

            if LIPSITZ_WEIGHTS:                    
                [p.data.clamp_(clamp_lower, clamp_upper) for p in self.discriminator.parameters()]

        return loss.item()


    def train_epoch_gen(self, k_g, energy_b, mom_point_b):
        """
        Train generator for one epoch
        
            Parameters
            ----------
                k_g : int
                    Number of iterations to train for.
                
                energy_b : torch.Tensor
                    Batch of energy distributions.
                
                mom_point_b : torch.Tensor
                    Batch of concatenated particle momentum and entrance point.
                                
            Returns:
            ----------
                loss.item() : float
                    Loss on the last iteration of training.
        """

        for _ in range(k_g):
            noise = torch.randn(len(energy_b), self.noise_dim).to(self.device)
            energy_gen = self.generator(noise, mom_point_b)
            
            if self.INSTANCE_NOISE:
                energy_b   = add_instance_noise(energy_b,   self.device)
                energy_gen = add_instance_noise(energy_gen, self.device)
            
            loss = self.gan_loss.g_loss(self.discriminator(energy_gen, mom_point_b))
            self.g_optimizer.zero_grad()
            loss.backward()
            self.g_optimizer.step()

        return loss.item()

    def validation(self):
        self.generator.eval()
        self.discriminator.eval()

        with torch.no_grad():

            #val_predictions_dis, val_predictions_gen = [], []
            val_energy_b, val_energy_gen  = [], []

            for energy_b, mom_b, point_b, pdg_b in self.valid_dataloader:
                energy_b, mom_b = energy_b.to(self.device), mom_b.to(self.device)
                point_b,  pdg_b = point_b .to(self.device), pdg_b.to(self.device)
                mom_point_b = torch.cat([mom_b.to(self.device), point_b.to(self.device)], dim=1)

                noise = torch.randn(len(energy_b), self.noise_dim).to(self.device)
                energy_gen = self.generator(noise, mom_point_b)
                
                val_energy_b.append(energy_b.detach().cpu().numpy())
                val_energy_gen.append(energy_gen.detach().cpu().numpy())

                #pred_dis_list = list(self.discriminator(energy_b,   mom_point_b).detach().cpu().numpy().ravel())
                #pred_gen_list = list(self.discriminator(energy_gen, mom_point_b).detach().cpu().numpy().ravel())
                
                #val_predictions_dis.append(pred_dis_list)
                #val_predictions_gen.append(pred_gen_list)

            val_energy_gen = np.concatenate(val_energy_gen, axis=0)
            val_energy_b   = np.concatenate(val_energy_b,   axis=0)

            self.val_prd_auc.append(prd_auc(val_energy_gen, val_energy_b, 
                                            self.VALID_SIZE))
        return

    def plot_training_progress(self, epoch, example_real, example_gen):
        """
        Plot training curves (loss, discrimination quality, prd-auc) and draw
        real/generated energy distributions and showers.
        
            Parameters
            ----------
                epoch : int
                    Current epoch number.
                
                example_real : torch.Tensor 30x30x1
                    Random energy distribution from the training dataset.
                
                example_gen : torch.Tensor
                    Generated energy distribution for the emaple from the 
                    training dataset. Real distribution is passed in example_real.
        """

        clear_output()

        print('Epoch #%d\nMean discriminator output on real data = %g\n'\
              'Mean discriminator output on generated data = %g' % 
              (epoch, np.mean(self.predictions_dis[-1]), np.mean(self.predictions_gen[-1])))
        
        fs_title = 14
        fs_axis = 12
        fs_ticks = 10

        f, ax = plt.subplots(1,3, figsize=(20, 5))
        ax[0].set_title ('Training loss', fontsize=fs_title)
        ax[0].set_xlabel('Epoch',     fontsize=fs_axis)
        ax[0].set_ylabel('Loss',          fontsize=fs_axis)

        ax[0].set_xticks     (np.linspace(0.0, len(self.dis_epoch_loss), 5))
        ax[0].set_xticklabels(np.linspace(0.0, epoch, 5))
        ax[0].plot(self.dis_epoch_loss, label='discriminator', color = 'red',  alpha=0.5)
        ax[0].plot(self.gen_epoch_loss, label='generator',     color = 'blue', alpha=0.5)
        ax[0].legend()

        ax[1].set_title ('Discrimination quality', fontsize=fs_title)
        ax[1].set_xlabel('Discriminator output',   fontsize=fs_axis)
        ax[1].set_ylabel('Number of examples',     fontsize=fs_axis)
        # get the bin edges to get an equal bin size for both distributions
        bins = np.histogram(np.hstack((self.predictions_dis[-1],
                                    self.predictions_gen[-1])), bins=100)[1]
        ax[1].hist(self.predictions_dis[-1], bins, label='real',      color = 'red',  ec='darkred',  alpha=0.5)
        ax[1].hist(self.predictions_gen[-1], bins, label='generated', color = 'blue', ec='darkblue', alpha=0.5)
        ax[1].legend()

        ax[2].set_xticks     (np.linspace(0.0, epoch, 5))
        ax[2].set_xticklabels(np.linspace(0.0, epoch, 5))
        ax[2].set_title ('PRD AUC', fontsize=fs_title)
        ax[2].set_xlabel('Epoch',   fontsize=fs_axis)
        ax[2].set_ylabel('PRD AUC', fontsize=fs_axis)
        #ax[2].set_xticks(np.arange(0,epoch))
        ax[2].plot(self.prd_auc,     label="Training",  color='red')
        ax[2].plot(self.val_prd_auc, label="Validaion", color='blue')
        ax[2].legend()

        #plot_energy_distr_real_generated(example_real, example_gen)
        #plot_shower_real_generated(example_real, example_gen) 
        plot_energy_and_shower(example_real, example_gen)
        plt.show()
        
        return