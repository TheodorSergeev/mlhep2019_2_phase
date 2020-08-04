import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as tck

def plot_energy_distr_real_generated(energ_distr_real, 
                                     energ_distr_generated):
    f, ax = plt.subplots(1,2, figsize=(10, 4))
    f.suptitle("Energy distribution", fontsize=16, y=.9)

    vmin = torch.max(energ_distr_real) 
    vmin = min(vmin, torch.min(energ_distr_generated))
    vmax = torch.max(energ_distr_real) 
    vmax = max(vmax, torch.max(energ_distr_generated))

    for i, title, en_dep in zip([0,1], ["Real", "Generated"],
                                [energ_distr_real[0], energ_distr_generated[0]]):
        ax[i].set_title (title,  fontsize=14)
        ax[i].set_xlabel(r'$x$', fontsize=12)
        ax[i].set_ylabel(r'$y$', fontsize=12)

        ax[i].set_xticks(np.arange(-0.5, 30., 1.), minor=True)
        ax[i].set_xticks(np.arange(-0.5, 30., 5.), minor=False)                    
        ax[i].set_xticklabels(np.arange(-15, 16, 5), minor=False, fontsize=10)

        ax[i].set_yticks(np.arange(-0.5, 30., 1.), minor=True)
        ax[i].set_yticks(np.arange(-0.5, 30., 5.), minor=False)                    
        ax[i].set_yticklabels(np.arange(-15, 16, 5), minor=False, fontsize=10)

        ax[i].grid(which='both', color='dimgray', linestyle='-', linewidth=0.5)
        im = ax[i].imshow(en_dep, origin = 'lower', cmap="inferno", 
                          vmin=vmin)#, vmax=vmax)

        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.subplots_adjust(wspace=0.1, hspace=0.05, top = 0.75)
    #plt.show()
    return


def plot_shower_real_generated(energ_distr_real, 
                               energ_distr_generated):
    f, ax = plt.subplots(1,2, figsize=(10, 4))
    f.suptitle("Shower", fontsize=16, y=.9)

    for i, title, en_dep in zip([0,1], ["Real", "Generated"],
                                [energ_distr_real[0], energ_distr_generated[0]]):
        ax[i].set_title (title,  fontsize=14)
        ax[i].set_xlabel(r'Cluster traverse width ($x$)', fontsize=12)
        ax[i].set_ylabel(r'Arbitrary units (sum along $y$)', fontsize=12)

        ax[i].set_xticks(np.arange(-0.5, 30., 1.), minor=True)
        ax[i].set_xticks(np.arange(-0.5, 30., 5.), minor=False)                    
        ax[i].set_xticklabels(np.arange(-15, 16, 5), minor=False, fontsize=10)

        ax[i].bar(np.arange(0,30), torch.sum(en_dep, axis=0), 
                  width=1.0, align='edge', edgecolor='black')

    plt.subplots_adjust(hspace=0.05, top = 0.75)
    return


def plot_energy_and_shower(energ_distr_real, energ_distr_generated):
    f, ax = plt.subplots(2,2, figsize=(8, 8))
    #f.suptitle("Energy distribution", fontsize=16, y=.9)

    vmin = torch.max(energ_distr_real) 
    vmin = min(vmin, torch.min(energ_distr_generated))
    vmax = torch.max(energ_distr_real) 
    vmax = max(vmax, torch.max(energ_distr_generated))

    for i, distr_type, en_dep in zip([0,1], ["Real", "Generated"],
                                [energ_distr_real[0], energ_distr_generated[0]]):
        for j, distr_name in zip([0,1], ["energy distribution", "shower"]):
            ax[j][i].set_title (distr_type + " " + distr_name,  fontsize=14)
            ax[j][i].set_xticks(np.arange(-0.5, 30., 1.), minor=True)
            ax[j][i].set_xticks(np.arange(-0.5, 30., 5.), minor=False)                    
            ax[j][i].set_xticklabels(np.arange(-15, 16, 5), minor=False, fontsize=10)

        ax[0][i].set_xlabel(r'$x$', fontsize=12)
        ax[0][i].set_ylabel(r'$y$', fontsize=12)
        ax[1][i].set_xlabel(r'Cluster traverse width ($x$)', fontsize=12)
        ax[1][i].set_ylabel(r'Arbitrary units (sum along $y$)', fontsize=12)

        ax[0][i].set_yticks(np.arange(-0.5, 30., 1.), minor=True)
        ax[0][i].set_yticks(np.arange(-0.5, 30., 5.), minor=False)                    
        ax[0][i].set_yticklabels(np.arange(-15, 16, 5), minor=False, fontsize=10)

        # plot shower
        ax[1][i].bar(np.arange(0,30), torch.sum(en_dep, axis=0), 
                  width=1.0, align='edge', edgecolor='black')

        # plot energy distribution
        ax[0][i].grid(which='both', color='dimgray', linestyle='-', linewidth=0.5)
        im = ax[0][i].imshow(en_dep, origin = 'lower', cmap="inferno", 
                          vmin=vmin)#, vmax=vmax)

        # add colorabar for the energy distributions
        divider = make_axes_locatable(ax[0][i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    #plt.show()
    return