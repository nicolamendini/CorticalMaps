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

# PARAMETERS TO BE SIMULATED
scalevals = torch.linspace(0,2.1,4)
scalevals[0] = 0.01
kvals = [1,2,3]
reps = 1
print(scalevals)
trials = len(scalevals)
ktrials = len(kvals)

# Function to collect stats
def run_print_stats():
        
    max_grid_size = round(config.CROPSIZE*kvals[-1]/config.KAPPA)
    max_grid_size = evenise(max_grid_size)
    affinities = torch.ones(reps,ktrials,trials)
    compressib = torch.ones(reps,ktrials,trials)
    maps = -torch.ones(reps,ktrials,trials,max_grid_size,max_grid_size)
    c = 3
    X = import_norb().type(torch.float16).cuda()[:,:,:162,0,c:-c,c:-c]
    X = X.reshape(-1,1,X.shape[-1], X.shape[-1])
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

                model, stats = run(X, bar=True, eval_at_end=config.EVALUATE)

                affinities[r,k,i] = stats['rf_affinity']
                compressib[r,k,i] = stats['avg_reco']

                start = (max_grid_size - config.GRID_SIZE)//2
                maps[r,k,i,start:start+config.GRID_SIZE,start:start+config.GRID_SIZE] = stats['map_tracker'][-1]
                
    sim_data = {'affinities': affinities, 'compressib': compressib, 'maps': maps}
    torch.save(sim_data, 'sim_data/gcal_compressib/files_to_plot/sim_data.pt')
    plot_from_file()
    
    os.system("shutdown -h 0")
    
# function to collect stats about noise robustness
def run_noise_stats():
    
    noise_res = torch.ones(trials, len(config.NOISE))
    fix_noise_res = torch.ones(trials, len(config.NOISE))
    sparse_res = torch.ones(trials, len(config.SPARSITY))
    rf_sparse_res = torch.ones(trials, len(config.RF_SPARSITY))
    mixed_res = torch.ones(trials, len(config.NOISE))
    maps = torch.ones(trials, config.GRID_SIZE, config.GRID_SIZE)
    
    c = 3
    X = import_norb().type(torch.float16).cuda()[:,:,:162,0,c:-c,c:-c]
    X = X.reshape(-1,1,X.shape[-1], X.shape[-1])
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
    
    for i in range(trials):
        
        config.EXC_STD = scalevals[i]
        
        model, stats = run(X, bar=True, eval_at_end=config.EVALUATE)
        
        noise_res[i] = stats['avg_noise']
        fix_noise_res[i] = stats['avg_fixed_noise']
        sparse_res[i] = stats['avg_sparsity']
        rf_sparse_res[i] = stats['avg_rf_sparsity']
        mixed_res[i] = stats['avg_mixed_case']
        maps[i] = stats['map_tracker'][-1]
        
    sim_data = {
        'avg_noise': noise_res,
        'avg_fixed_noise': fix_noise_res,
        'avg_sparsity': sparse_res,
        'avg_rf_sparsity': rf_sparse_res,
        'avg_mixed_case': mixed_res,
        'maps' : maps
    }
    
    torch.save(sim_data, 'sim_data/noise_tests/files_to_plot/noise_data.pt')
    plot_noise_data()
    os.system("shutdown -h 0")
    
def plot_lambda():
    
    sim_data = torch.load('sim_data/noise_tests/files_to_plot/noise_data.pt')
    maps = sim_data['maps']
    for i in range(trials):
        avg_peak, _, _ = get_typical_dist_fourier(
                    maps[i], 
                    0, 
                    mask=0, 
                    match_std=1
                )
        plt.imshow(maps[i], cmap='hsv')
        plt.show()
        print(avg_peak)
    
    
# plot the results of the noise simulations
def plot_noise_data():
    
    sim_data = torch.load('sim_data/noise_tests/files_to_plot/noise_data.pt')
    noise_res = sim_data['avg_noise']
    fix_noise_res = sim_data['avg_fixed_noise']
    sparse_res = sim_data['avg_sparsity']
    rf_sparse_res = sim_data['avg_rf_sparsity']
    mixed_res = sim_data['avg_mixed_case']

    stn = 10 * torch.log10(0.25 / torch.tensor(config.NOISE)**2)
    
    #print(noise_res.shape, fix_noise_res.shape, sparse_res.shape, rf_sparse_res.shape, mixed_res.shape, stn.shape)
    fs = 16
    #plt.title('Robustness of V1 to noise, for different σ conditions',  fontsize=fs)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.subplots_adjust(wspace=0.12, hspace=0.3)
    plot_var = noise_res
    plt.xlabel('signal to noise ratio (dB)', fontsize=fs)
    plt.ylabel('V1 output stability', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    for i in range(plot_var.shape[0]):
        plt.plot(stn, plot_var[i])
    plt.subplot(1,2,2)
    plot_var = fix_noise_res
    plt.xlabel('signal to noise ratio (dB)', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    for i in range(plot_var.shape[0]):
        plt.plot(stn, plot_var[i])
    plt.savefig('sim_data/noise_tests/noise.png', bbox_inches='tight')
    plt.clf()
    
    fs = 16
    #plt.title('Robustness of V1 to connection sparsity, for different σ conditions',  fontsize=fs)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.ylim(0.1,1.05)
    plt.subplots_adjust(wspace=0.12, hspace=0.3)
    plot_var = rf_sparse_res
    plt.xlabel('connection density (%)', fontsize=fs)
    plt.ylabel('V1 output stability', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    for i in range(plot_var.shape[0]):
        plt.plot(config.SPARSITY, plot_var[i])
    plt.subplot(1,2,2)
    plt.ylim(0.1,1.05)
    plot_var = sparse_res
    plt.xlabel('connection density (%)', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    for i in range(plot_var.shape[0]):
        plt.plot(config.SPARSITY, plot_var[i])
    plt.savefig('sim_data/noise_tests/sparsity.png', bbox_inches='tight')
    plt.clf()
    
    plt.figure(figsize=(6,4))
    plot_var = mixed_res
    #plt.title('Noise robustness with afferent and lateral sparsity',  fontsize=fs)
    plt.xlabel('signal to noise ratio (dB)', fontsize=fs)
    plt.ylabel('V1 output stability', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    for i in range(plot_var.shape[0]):
        plt.plot(stn, plot_var[i])
    #ax.plot_surface(scalevals[:,None], X[None,:], sim_data)
    plt.savefig('sim_data/noise_tests/mixed_case.png', bbox_inches='tight')
    plt.clf()

    
# function to plot the results of many trials with different scale parameters
def plot_from_file():
    
    sim_data = torch.load('sim_data/gcal_compressib/files_to_plot/sim_data.pt')
    
    affinities = sim_data['affinities']
    compressib = sim_data['compressib']
    maps = sim_data['maps']
    
    max_grid_size = maps.shape[-1]
    spectra = torch.zeros(reps,ktrials,trials,max_grid_size-1,max_grid_size-1)
    ratios = torch.ones(reps,ktrials,trials)
    avg_peaks = torch.ones(reps,ktrials,trials)
    counts = torch.ones(reps,ktrials,trials)
    hists = torch.ones(reps,ktrials,trials,max_grid_size//2-1)
    
    for r in range(reps):
        for k in range(ktrials):
            for i in tqdm(range(trials)):
                
                config.EXC_STD = scalevals[i]
                config.EXPANSION = kvals[k]/config.KAPPA
                config.DILATION = kvals[k]
                config.COMPRESSION = kvals[k]
                config.GRID_SIZE = round(config.CROPSIZE*config.EXPANSION)
                config.GRID_SIZE = evenise(config.GRID_SIZE)
                config.MAPCHOP = round(kvals[k]*2)
                
                start = (max_grid_size - config.GRID_SIZE)//2
                count, pinwheels, pinwheels_copy = count_pinwheels(
                    maps[r,k,i,start:start+config.GRID_SIZE,start:start+config.GRID_SIZE], 
                    config.MAPCHOP
                )
                avg_peak, spectrum, avg_hist = get_typical_dist_fourier(
                    maps[r,k,i], 
                    0, 
                    mask=2*kvals[-1]/kvals[k], 
                    smoothing_std=config.GRID_SIZE-config.MAPCHOP*2,
                    match_std=kvals[-1]/kvals[k]
                )
                ratio = count / (config.GRID_SIZE-config.MAPCHOP*2)**2 * avg_peak**2       
                spectra[r,k,i] = spectrum
                ratios[r,k,i] = ratio
                counts[r,k,i] = count
                hists[r,k,i] = avg_hist
    
    #sns.set()    
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
    rings = best_rings.mean(0)**2
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
    #best_idxs = ensemble.mean(0).max(1,keepdim=True)[1]
    matches = [match_ring(rings[i], len(kvals)/kvals[i]) for i in range(len(kvals))]
    lambda_max = torch.tensor([1/match[1] for match in matches])[:,None]
    #best_idxs = best_idxs[:,:,None].expand(-1,-1,max_grid_size//2-1)
    #hists_max = hists.mean(0).gather(1,best_idxs)
    hists_max = torch.cat([match[0][None] for match in matches])[:,None]
    hists_max /= hists_max.max(2,keepdim=True)[0]
    ticks_range = (-0.5,0.5,-0.5,0.5)
    lgn_size = round(max_grid_size/kvals[-1]*config.KAPPA)
    start = round(max_grid_size - lgn_size)//2
    for k in range(ktrials):

        pad = (max_grid_size - evenise(round((config.CROPSIZE)*kvals[k]/config.KAPPA)) + config.MAPCHOP)//2
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
        style = 'k--'
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
        #ring[~freq_mask] = torch.tensor(1)/0.
        im=ax.imshow(rings[k], cmap='Greys', extent=ticks_range)
        ax.plot(np.linspace(0,0.5,max_grid_size//2-1), hists_max[k,0]/4-1/2, linewidth=2, color='red')
        ax.plot([1/lambda_max[k]]*2, [-0.5, 0.], linewidth=2, linestyle=':', color='red')
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
        markers_on = ensemble_mean[k].argmax()
        plt.plot(scalevals*mm_conv, means, color='black', markevery=[markers_on], marker='o')
        plt.fill_between(scalevals*mm_conv, means - stds, means + stds, color='grey', alpha=0.2)
        plt.plot(scalevals*mm_conv, [1]*len(scalevals), color='grey', linestyle='--', linewidth=2)
    
    
    plt.savefig('sim_data/gcal_compressib/tradeoff.png')
    plt.savefig('sim_data/gcal_compressib/tradeoff.svg')
    
    
    