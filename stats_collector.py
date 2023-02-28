# Author: Nicola Mendini
# Code used to collect some stats about the Affinity/Compressibility trade-off

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
import math
import torchvision.transforms.functional as TF

import config
from training_loop import *
from data_imports import *
from retina_lgn import *
from maps_helper import *

# Function to collect stats
def run_print_stats(scalevals, kvals, X):
        
    trials = len(scalevals)
    ktrials = len(kvals)

    max_grid_size = round(config.CROPSIZE*kvals[-1]/1.5)
    max_grid_size = evenise(max_grid_size)
    model = None
    affinities = torch.ones(ktrials,trials)
    compressib = torch.ones(ktrials,trials)
    maps = -torch.ones(ktrials,trials,max_grid_size,max_grid_size)
    spectra = torch.zeros(ktrials,trials,max_grid_size+1,max_grid_size+1)
    ratios = torch.ones(ktrials,trials)
    avg_peaks = torch.ones(ktrials,trials)

    for k in range(ktrials):
        for i in range(trials):

            config.EXC_STD = scalevals[i]
            config.SCALE = int(torch.round((config.EXC_STD)*5))
            config.SCALE = oddenise(config.SCALE)
            config.EXPANSION = kvals[k]/1.5
            config.DILATION = kvals[k]
            config.COMPRESSION = kvals[k]
            config.GRID_SIZE = round(config.CROPSIZE*config.EXPANSION)
            config.GRID_SIZE = evenise(config.GRID_SIZE)

            if config.COMPRESSION==1:
                config.RECO = False
            else:
                config.RECO = True

            model, stats = run(X, bar=True)

            affinities[k,i] = stats['rf_affinity']
            
            if config.COMPRESSION>1:
                compressib[k,i] = stats['avg_reco']
                
            start = (max_grid_size - config.GRID_SIZE)//2
            maps[k,i,start:start+config.GRID_SIZE,start:start+config.GRID_SIZE] = stats['map_tracker'][-1]
            
            count, pinwheels, pinwheels_copy = count_pinwheels(maps[k,i], config.MAPCHOP, thresh=5.3)
            avg_peak, spectrum, avg_hist =  get_typical_dist_fourier(maps[k,i], config.MAPCHOP, mask=1)
            ratio = count / (config.GRID_SIZE-config.MAPCHOP*2)**2 * avg_peak**2
            
            spectra[k,i,1:,1:] = spectrum
            ratios[k,i] = ratio
            avg_peaks[k,i] = avg_peak

    ensemble = affinities.mean(0, keepdim=True) * compressib    
    print(ensemble, affinities, compressib, ratios, avg_peaks)    
    return ensemble, affinities, compressib, maps, ratios, avg_peaks, spectra

# function to plot the results of many trials with different scale parameters
def plot_results(kvals, scalevals, ensemble, affinities, compressib, maps, ratios, avg_peaks, spectra):
    
    trials = len(scalevals)
    ktrials = len(kvals)
    ensemble_err = affinities.std(0)
    best_maps_idxs = ensemble.max(1, keepdim=True)[1][:,:,None,None]
    best_maps_idxs = best_maps_idxs.expand(-1,-1,maps.shape[-1],maps.shape[-1])
    best_maps = maps.gather(1,best_maps_idxs)
    best_rings = spectra[:,:,1:,1:].gather(1,best_maps_idxs)
    max_grid_size = round(config.CROPSIZE*kvals[-1]/1.5)
    max_grid_size = evenise(max_grid_size)
    
    plt.figure(figsize=(ktrials*5,15))
    plt.suptitle('Trade-Off when varying Map Scale, for ' + str(ktrials) + ' different sparsity conditions', fontsize=30)

    for k in range(ktrials):

        plt.subplot(3,ktrials,1+k)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        if k==0:
            plt.ylabel('Ensemble Score', fontsize=20)
        plt.xlabel('Map Scale', fontsize=20)
        plt.title('K = '+ str(k+1), fontsize=20)
        plt.ylim(0.60,0.85)
        plt.errorbar(scalevals, ensemble[k], 2*ensemble_err)

    for k in range(ktrials):

        map_to_plot = best_maps[k,0]
        map_to_plot[map_to_plot==-1] = torch.tensor(1.)/0.
        plt.subplot(3,ktrials,ktrials+1+k)
        plt.axis('off')
        plt.imshow(map_to_plot, cmap='hsv')
        
    for k in range(ktrials):

        plt.subplot(3,ktrials,ktrials*2+1+k)
        plt.axis('off')
        plt.imshow(best_rings[k,0])

    plt.savefig('sim_data/gcal_compressib/tradeoff.png')
    plt.savefig('sim_data/gcal_compressib/tradeoff.svg')
    
    plt.figure(figsize=(10,5))
    plt.suptitle('Pinwheel Density and Typical Domain Distance vs Map Scale', fontsize=20)
    plt.subplot(1,2,1)
    plt.ylabel('Pinwheel Density', fontsize=14)
    plt.xlabel('Map Scale', fontsize=14)
    plt.errorbar(scalevals, ratios.mean(0), ratios.std(0))
    plt.subplot(1,2,2)
    plt.ylabel('Typical Distance', fontsize=14)
    plt.xlabel('Map Scale', fontsize=14)
    plt.errorbar(scalevals, avg_peaks.mean(0), avg_peaks.std(0))
    
    plt.savefig('sim_data/gcal_compressib/density.png')
    plt.savefig('sim_data/gcal_compressib/density.svg')
    
# function to start the collection of statistics for different parameter configurations
def collect():
    
    X = import_norb().type(torch.float16).cuda()
    X = X[:,:,:162,0,:,:].reshape(-1,1,config.INPUT_SIZE,config.INPUT_SIZE)
    #X = import_fruit()
    X, X_orig, X_mask = retina_lgn(
        X,
        config.RET_LOG_STD,
        config.LGN_LOG_STD,
        config.HARDGC,
        config.K_STD,
        config.CG_EPS,
        config.SATURATION,
        config.LGN_ITERS
    )
    X = X.float()
    
    scalevals = torch.linspace(0.5,3,6)
    scalevals[0] = 0.1
    kvals = [1,2,3,4,5]
    print(scalevals)

    ensemble, affinities, compressib, maps, ratios, avg_peaks, spectra = run_print_stats(scalevals, kvals,X)
    plot_results(kvals, scalevals, ensemble, affinities, compressib, maps, ratios, avg_peaks, spectra)

    torch.save(ensemble, 'sim_data/gcal_compressib/ensemble.pt')
    torch.save(affinities, 'sim_data/gcal_compressib/affinities.pt')
    torch.save(compressib, 'sim_data/gcal_compressib/compressib.pt')
    torch.save(maps, 'sim_data/gcal_compressib/maps.pt')
    torch.save(ratios, 'sim_data/gcal_compressib/ratios.pt')
    torch.save(avg_peaks, 'sim_data/gcal_compressib/avg_peaks.pt')
    torch.save(spectra, 'sim_data/gcal_compressib/spectra.pt')