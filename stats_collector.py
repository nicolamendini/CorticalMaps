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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import config
from training_loop import *
from data_imports import *
from retina_lgn import *
from maps_helper import *

# Function to collect stats
def run_print_stats(scalevals, kvals, X, reps=3):
        
    trials = len(scalevals)
    ktrials = len(kvals)

    max_grid_size = round(config.CROPSIZE*kvals[-1]/1.5)
    max_grid_size = evenise(max_grid_size)
    model = None
    affinities = torch.ones(reps,ktrials,trials)
    compressib = torch.ones(reps,ktrials,trials)
    maps = -torch.ones(reps,ktrials,trials,max_grid_size,max_grid_size)
    spectra = torch.zeros(reps,ktrials,trials,max_grid_size-1,max_grid_size-1)
    ratios = torch.ones(reps,ktrials,trials)
    avg_peaks = torch.ones(reps,ktrials,trials)
    counts = torch.ones(reps,ktrials,trials)
    hists = torch.ones(reps,ktrials,trials,max_grid_size//2-1)

    for r in range(reps):
        for k in range(ktrials):
            for i in range(trials):

                config.EXC_STD = scalevals[i]
                config.EXC_SCALE = int(torch.round((config.EXC_STD)*5))
                config.EXC_SCALE = oddenise(config.EXC_SCALE)
                config.EXPANSION = kvals[k]/1.5
                config.DILATION = kvals[k]
                config.COMPRESSION = kvals[k]
                config.GRID_SIZE = round(config.CROPSIZE*config.EXPANSION)
                config.GRID_SIZE = evenise(config.GRID_SIZE)
                config.RF_SIZE = round(config.RF_STD*config.EXPANSION/round(config.EXPANSION)*5)
                config.RF_SIZE = oddenise(config.RF_SIZE)
                # INH_SCALE is constant
                #config.INH_SCALE = round(config.RF_STD*config.EXPANSION/config.DILATION*5)
                #config.INH_SCALE = oddenise(config.INH_SCALE)

                if config.COMPRESSION==1:
                    config.RECO = False
                else:
                    config.RECO = True

                model, stats = run(X, bar=True)

                affinities[r,k,i] = stats['rf_affinity']

                if config.COMPRESSION>1:
                    compressib[r,k,i] = stats['avg_reco']

                start = (max_grid_size - config.GRID_SIZE)//2
                maps[r,k,i,start:start+config.GRID_SIZE,start:start+config.GRID_SIZE] = stats['map_tracker'][-1]

                count, pinwheels, pinwheels_copy = count_pinwheels(stats['map_tracker'][-1], config.MAPCHOP)
                avg_peak, spectrum, avg_hist = get_typical_dist_fourier(
                    maps[r,k,i], 
                    0, 
                    mask=kvals[-1]/kvals[k]+1, 
                    smoothing_std=config.GRID_SIZE
                )
                ratio = count / (config.GRID_SIZE-config.MAPCHOP*2)**2 * avg_peak**2            
                spectra[r,k,i] = spectrum
                ratios[r,k,i] = ratio
                avg_peaks[r,k,i] = avg_peak
                counts[r,k,i] = count
                hists[r,k,i] = avg_hist
        
    print(affinities, compressib, ratios, avg_peaks, counts)    
    return affinities, compressib, maps, ratios, avg_peaks, spectra, hists, counts

# function to plot the results of many trials with different scale parameters
def plot_results(kvals, scalevals, affinities, compressib, maps, ratios, avg_peaks, spectra, hists):
    
    trials = len(scalevals)
    ktrials = len(kvals)
    ensemble = affinities * compressib
    ensemble_mean = ensemble.mean(0)
    ensemble_err = ensemble.std(0)
    best_map_idxs_raw = ensemble.permute(0,2,1).reshape(-1,ktrials).max(0, keepdim=True)[1]
    best_map_idxs_raw = best_map_idxs_raw[:,:,None,None]
    
    # selecting the best maps
    best_map_idxs = best_map_idxs_raw.expand(-1,-1,maps.shape[-1],maps.shape[-1])
    best_maps = maps.permute(0,2,1,3,4).reshape(-1,ktrials,maps.shape[-1],maps.shape[-1]).gather(0,best_map_idxs)
    
    # selecting the best spectra
    best_map_idxs = best_map_idxs_raw.expand(-1,-1,spectra.shape[-1],spectra.shape[-1])
    best_rings = spectra.permute(0,2,1,3,4).reshape(-1,ktrials,spectra.shape[-1],spectra.shape[-1])\
        .gather(0,best_map_idxs)
    max_grid_size = round(config.CROPSIZE*kvals[-1]/1.5)
    max_grid_size = evenise(max_grid_size)
    
    rows = 4
    
    fig = plt.figure(figsize=(ktrials*5,rows*5))
    plt.suptitle('Finding the optimal Map Scale, for ' + str(ktrials) + ' different sparsity conditions K', fontsize=30)

    for k in range(ktrials):

        plt.subplot(rows,ktrials,1+k)
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.subplots_adjust(hspace=0.4)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.locator_params(nbins=6)
        if k==0:
            plt.ylabel('Ensemble Score', fontsize=20)
            plt.xlabel('Map Scale', fontsize=20)
        plt.title('K = '+ str(kvals[k]), fontsize=30)
        plt.ylim(0.6,1)
        plt.errorbar(scalevals, ensemble_mean[k], 2*ensemble_err[k], color='black')
        
    for k in range(ktrials):

        plt.subplot(rows,ktrials,ktrials*1+1+k)
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.locator_params(nbins=6)
        if k==0:
            plt.ylabel('Pinwheel Density', fontsize=20)
            plt.xlabel('Map Scale', fontsize=20)
        plt.ylim(0,10)
        plt.errorbar(scalevals, ratios[:,k].mean(0), 2*ratios[:,k].std(0), color='black')
        plt.plot(scalevals, [np.pi]*len(scalevals), linewidth=2, color='red')

    ref_ax=0
    best_idxs = ensemble.mean(0).max(1,keepdim=True)[1]
    lambda_max = avg_peaks.mean(0).gather(1,best_idxs)
    best_idxs = best_idxs[:,:,None].expand(-1,-1,max_grid_size//2-1)
    hists_max = hists.mean(0).gather(1,best_idxs)
    hists_max /= hists_max.max(2,keepdim=True)[0]
    ticks_range = (-0.5,0.5,-0.5,0.5)
    for k in range(ktrials):

        map_to_plot = best_maps[0,k]
        map_to_plot[map_to_plot==-1] = torch.tensor(1.)/0.
        plt.subplot(rows,ktrials,ktrials*2+1+k)
        plt.locator_params(nbins=6)
        #plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        if k==0:
            plt.ylabel('Size (neurons)', fontsize=20)
            plt.xlabel('Size (neurons)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.locator_params(nbins=5)
        plt.imshow(map_to_plot, cmap='hsv')
        ref_ax = plt.gca()
        fontprops = fm.FontProperties(size=20)
        scalebar = AnchoredSizeBar(
                            ref_ax.transData,
                            lambda_max[k], '1 Λ', 'upper right', 
                            pad=0.4,
                            color='black',
                            frameon=True,
                            size_vertical=1,
                            fontproperties=fontprops,
                            borderpad=0.3,
                            sep=5      
                        )
        ref_ax.add_artist(scalebar)
        
    for k in range(ktrials):

        plt.subplot(rows,ktrials,ktrials*3+1+k)
        plt.locator_params(nbins=6)
        #plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        if k==0:
            plt.ylabel('Frequency (1/neurons)', fontsize=20)
            plt.xlabel('Frequency (1/neurons)', fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax = plt.gca()
        ax.set_box_aspect(1)
        ax.imshow(best_rings[0,k]**2, cmap='Greys', extent=ticks_range)
        ax.plot(np.linspace(0,0.5,max_grid_size//2-1), hists_max[k,0]/4-0.5, linewidth=2, color='red')
    
    
    plt.savefig('sim_data/gcal_compressib/tradeoff.png')
    plt.savefig('sim_data/gcal_compressib/tradeoff.svg')
    
    # Pinwheel density plot
    """
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
    """
    
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
        config.RF_STD,
        config.CG_EPS,
        config.SATURATION,
        config.LGN_ITERS
    )
    X = X.float()
    
    scalevals = torch.linspace(0,3,16)
    scalevals[0] = 0.1
    kvals = [1,2,3,4]
    print(scalevals)

    affinities, compressib, maps, ratios, avg_peaks, spectra, hists, counts = run_print_stats(scalevals, kvals,X)
    plot_results(kvals, scalevals, affinities, compressib, maps, ratios, avg_peaks, spectra, hists)

    torch.save(affinities, 'sim_data/gcal_compressib/affinities.pt')
    torch.save(compressib, 'sim_data/gcal_compressib/compressib.pt')
    torch.save(maps, 'sim_data/gcal_compressib/maps.pt')
    torch.save(ratios, 'sim_data/gcal_compressib/ratios.pt')
    torch.save(avg_peaks, 'sim_data/gcal_compressib/avg_peaks.pt')
    torch.save(spectra, 'sim_data/gcal_compressib/spectra.pt')
    torch.save(hists, 'sim_data/gcal_compressib/hists.pt')
    torch.save(counts, 'sim_data/gcal_compressib/counts.pt')