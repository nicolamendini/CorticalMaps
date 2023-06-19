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
import os
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import config
from training_loop import *
from data_imports import *
from retina_lgn import *
from maps_helper import *
from scipy.optimize import curve_fit

# Function to collect stats
def run_print_stats(scalevals, kvals, X, reps=5):
        
    trials = len(scalevals)
    ktrials = len(kvals)

    max_grid_size = round(config.CROPSIZE*kvals[-1]/config.KAPPA)
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
                config.EXPANSION = kvals[k]/config.KAPPA
                config.DILATION = kvals[k]
                config.COMPRESSION = kvals[k]
                config.GRID_SIZE = round(config.CROPSIZE*config.EXPANSION)
                config.GRID_SIZE = evenise(config.GRID_SIZE)
                config.MAPCHOP = kvals[k]*3

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
                    mask=2.5*kvals[-1]/kvals[k], 
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
    
    #sns.set()
    
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
    max_grid_size = round(config.CROPSIZE*kvals[-1]/config.KAPPA)
    max_grid_size = evenise(max_grid_size)
    
    rows = 4
    fs = 18
    
    fig = plt.figure(figsize=(ktrials*5,rows*5))
    plt.subplots_adjust(
                    left=0.1,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    hspace=0.0
    )
    
    
    mm_conv = 0.05
    scale_ticks = (0, 0.05, 0.1, 0.15)
    for k in range(ktrials):

        plt.subplot(rows,ktrials,1+k)
        ax = plt.gca()
        ax.set_box_aspect(1)
        plt.xticks(fontsize=fs, ticks=scale_ticks, labels=scale_ticks)
        plt.yticks(fontsize=fs)
        plt.locator_params(nbins=6)
        if k==0:
            plt.ylabel('Transmission Accuracy', fontsize=fs)
        plt.xlabel('Excitation Range (mm)', fontsize=fs)
        best = format(float(scalevals[ensemble_mean[k].argmax()])*mm_conv, '.3f')
        plt.title('Projection Sparsity: '+ str(kvals[k]) + '\n' + 'Optimal Range: '+ best +'mm',  fontsize=fs)
        plt.ylim(0.6,1.01)
        means = ensemble_mean[k]
        stds = ensemble_err[k]
        markers_on = means.argmax()
        plt.plot(scalevals*mm_conv, affinities[:,k].mean(0), 'k:', label='LGN to V1')
        plt.plot(scalevals*mm_conv, compressib[:,k].mean(0), 'k--', label='V1 to V2')
        plt.plot(scalevals*mm_conv, means, color='black', label='combined', markevery=[markers_on], marker='o')
        plt.fill_between(scalevals*mm_conv, means - stds, means + stds, color='grey', alpha=0.2)
        plt.legend(fontsize=16, loc=(0.4,0.5), frameon=False)
        
    ref_ax=0
    best_idxs = ensemble.mean(0).max(1,keepdim=True)[1]
    lambda_max = avg_peaks.mean(0).gather(1,best_idxs)
    best_idxs = best_idxs[:,:,None].expand(-1,-1,max_grid_size//2-1)
    hists_max = hists.mean(0).gather(1,best_idxs)
    hists_max /= hists_max.max(2,keepdim=True)[0]
    ticks_range = (-0.5,0.5,-0.5,0.5)
    lgn_size = round(max_grid_size/kvals[-1]*config.KAPPA)
    start = round(max_grid_size - lgn_size)//2
    for k in range(ktrials):

        pad = (max_grid_size - evenise(round((config.CROPSIZE -3)*kvals[k]/config.KAPPA)))//2
        map_to_plot = best_maps[0,k]
        map_to_plot[map_to_plot==-1] = torch.tensor(1.)/0.
        map_to_plot[:pad] = torch.tensor(1.)/0.
        map_to_plot[-pad:] = torch.tensor(1.)/0.
        map_to_plot[:,:pad] = torch.tensor(1.)/0.
        map_to_plot[:,-pad:] = torch.tensor(1.)/0.
        plt.subplot(rows,ktrials,ktrials*1+1+k)
        plt.locator_params(nbins=6)
        #plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        if k==0:
            plt.ylabel('mm', fontsize=fs)
            plt.text(start,start-3,'LGN size', fontsize=fs)
        plt.xlabel('mm', fontsize=fs)
        plt.xticks(fontsize=fs, ticks=(0,50,100), labels=(-2.5, 0, 2.5))
        plt.yticks(fontsize=fs, ticks=(0,50,100), labels=(-2.5, 0, 2.5))
        plt.locator_params(nbins=3)
        #plt.box(False)
        #plt.tick_params(left=False, bottom=False)
        plt.imshow(map_to_plot, cmap='hsv')
        ref_ax = plt.gca()
        fontprops = fm.FontProperties(size=fs)
        lambda_plot = float(lambda_max[k]) * mm_conv
        scalebar = AnchoredSizeBar(
                            ref_ax.transData,
                            lambda_max[k], 'Λ = '+ format(lambda_plot, '.2f')+'mm', 'lower right', 
                            pad=0.4,
                            color='black',
                            frameon=True,
                            size_vertical=1,
                            fontproperties=fontprops,
                            borderpad=0.3,
                            sep=5      
                        )
        ref_ax.add_artist(scalebar)
        style = 'k-'
        plt.plot([start,start+lgn_size],[start,start], style, linewidth=2)
        plt.plot([start+lgn_size,start+lgn_size],[start,start+lgn_size], style, linewidth=2)
        plt.plot([start,start],[start+lgn_size,start], style, linewidth=2)
        plt.plot([start,start+lgn_size],[start+lgn_size,start+lgn_size], style, linewidth=2)
        
        
    #freq_mask = get_circle(best_rings.shape[-1], best_rings.shape[-1]/2)[0,0]
    for k in range(ktrials):

        plt.subplot(rows,ktrials,ktrials*2+1+k)
        plt.locator_params(nbins=3)
        #plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        if k==0:
            plt.ylabel('mm\u207B\u00B9', fontsize=fs)
        plt.xlabel('mm\u207B\u00B9', fontsize=fs)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        ax = plt.gca()
        ax.set_box_aspect(1)
        ring = best_rings[:,k].mean(0)**2
        #ring[~freq_mask] = torch.tensor(1)/0.
        im=ax.imshow(ring, cmap='Greys', extent=ticks_range)
        ax.plot(np.linspace(0,0.5,max_grid_size//2-1), hists_max[k,:].mean(0)/4-1/2, linewidth=2, color='red')
        ax.plot([1/lambda_max[k]]*2, [-0.5, 0.5], linewidth=2, linestyle=':', color='red')
        divider = make_axes_locatable(ax)
        cb = fig.colorbar(im,  cax=ax.inset_axes((0.9, 0.65, 0.05, 0.3)))
        cb.set_ticks([0, (best_rings[0,k]**2).max()])
        cb.set_ticklabels(['min', 'max'])
        cb.ax.tick_params(labelsize=fs)
        cb.ax.yaxis.set_ticks_position('left')
        
    for k in range(ktrials):

        plt.subplot(rows,ktrials,ktrials*3+1+k)
        ax = plt.gca()
        ax.set_box_aspect(1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
        plt.xticks(fontsize=fs, ticks=scale_ticks, labels=scale_ticks)
        plt.yticks(fontsize=fs, ticks=(0,1,2))
        if k==0:
            plt.ylabel('Pinwheel Density', fontsize=fs)
        plt.xlabel('Excitation Range (mm)', fontsize=fs)
        plt.ylim(0,2)
        means = ratios[:,k].mean(0)/np.pi
        stds = ratios[:,k].std(0)/np.pi
        plt.errorbar(scalevals*mm_conv, means, 0, color='black')
        plt.fill_between(scalevals*mm_conv, means - stds, means + stds, color='grey', alpha=0.2)
        plt.plot(scalevals*mm_conv, [1]*len(scalevals), color='grey', linestyle='--', linewidth=2)
    
    
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
    
    c = 3
    X = import_norb().type(torch.float16).cuda()[:,:,:162,0,c:-c,c:-c]
    input_size = X.shape[-1]
    X = X.reshape(-1,1,input_size, input_size)
    #X = F.pad(X, (input_size-1,0,input_size-1,0), mode='reflect')
    input_size = X.shape[-1]

    X, X_orig, X_mask = retina_lgn(
        X,
        config.RET_LOG_STD,
        config.LGN_LOG_STD,
        config.HARDGC,
        config.RF_STD,
        config.CG_EPS,
        config.CG_SATURATION,
        config.LGN_ITERS
    )

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
    
    os.system("shutdown -h 0")
    
def plot_from_file():
    
    affinities = torch.load('sim_data/gcal_compressib/files_to_plot/affinities.pt')
    compressib = torch.load('sim_data/gcal_compressib/files_to_plot/compressib.pt')
    maps = torch.load('sim_data/gcal_compressib/files_to_plot/maps.pt')
    ratios = torch.load('sim_data/gcal_compressib/files_to_plot/ratios.pt')
    avg_peaks = torch.load('sim_data/gcal_compressib/files_to_plot/avg_peaks.pt')
    spectra = torch.load('sim_data/gcal_compressib/files_to_plot/spectra.pt')
    hists = torch.load('sim_data/gcal_compressib/files_to_plot/hists.pt')
    counts = torch.load('sim_data/gcal_compressib/files_to_plot/counts.pt')
    
    plot_results(kvals, scalevals, affinities, compressib, maps, ratios, avg_peaks, spectra, hists)
    
    
scalevals = torch.linspace(0.2,3,15)
kvals = [1,2,3,4]
print(scalevals)
    
