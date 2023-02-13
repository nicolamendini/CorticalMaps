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

# Function to collect stats
def run_print_stats(scalevals, kvals, X):
        
    trials = len(scalevals)
    ktrials = len(kvals)

    model = None
    affinities = torch.ones(ktrials,trials)
    compressib = torch.ones(ktrials,trials)
    maps = torch.zeros(ktrials,trials,config.GRID_SIZE,config.GRID_SIZE)
    spectra = torch.zeros(ktrials,trials,config.GRID_SIZE+1,config.GRID_SIZE+1)
    ratios = torch.ones(ktrials,trials)
    avg_peaks = torch.ones(ktrials,trials)

    for k in range(ktrials):
        for i in range(trials):

            config.SCALE = scalevals[i]
            config.DILATION = max(int(config.SCALE*2),1)
            config.COMPRESSION = kvals[k]

            if config.COMPRESSION==1:
                config.RECO = False
            else:
                config.RECO = True

            model, stats = run(X, bar=True)

            affinities[k,i] = stats['rf_affinity']
            
            if config.COMPRESSION>1:
                compressib[k,i] = stats['avg_reco']
                
            maps[k,i] = stats['map_tracker'][-1]
            
            count, pinwheels, pinwheels_copy = count_pinwheels(maps[k,i], config.GRID_SIZE, config.MAPCHOP, thresh=5.3)
            avg_peak, spectrum, avg_hist = get_typical_dist_fourier(
                maps[k,i], config.GRID_SIZE, config.MAPCHOP, match_std=2)
            ratio = count / (config.GRID_SIZE-config.MAPCHOP*2)**2 * avg_peak**2
            
            spectra[k,i] = spectrum
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

        plt.subplot(3,ktrials,ktrials+1+k)
        plt.axis('off')
        plt.imshow(best_maps[k,0], cmap='hsv')
        
    for k in range(ktrials):

        plt.subplot(3,ktrials,ktrials*2+1+k)
        plt.axis('off')
        plt.imshow(best_rings[k,0])

    plt.savefig('sim_data/gcal_compressib/tradeoff.png')
    
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
    
# function to start the collection of statistics for different parameter configurations
def collect():
    
    X = import_norb().type(torch.float16).cuda()
    X = X[:,:,:162,0,:,:].reshape(-1,1,config.INPUT_SIZE,config.INPUT_SIZE)
    #X = import_fruit()
    X, X_orig, X_mask = retina_lgn(
        X,
        config.RET_LOG_STD,
        config.RET_STRENGTH,
        config.LGN_STRENGTH,
        config.RET_LOG_STD,
        config.HARDGC,
        config.CG_STD,
        config.CG_SCALER,
        config.CG_EPS,
        config.SAME_MEAN
    )
    X = X.float()
    
    scalevals = torch.linspace(0,3,11)
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