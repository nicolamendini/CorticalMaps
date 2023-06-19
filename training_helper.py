# Author: Nicola Mendini
# Helper functions useful to develop Models of Cortical Maps

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import random
import math
import torchvision.transforms.functional as TF

import config
from maps_helper import *

# initialise the dictionary containing stats about the model
def init_stats_dict(stats, device, crange, cropsize):
    
    if not stats:
        stats = {}
        stats['rf_affinity'] = 0
        stats['reco_rf_affinity'] = 0
        stats['avg_acts'] = torch.zeros((1,1,config.GRID_SIZE,config.GRID_SIZE), device=device) + config.TARGET_ACT
        stats['global_corr'] = torch.zeros(
            (config.CORR_SAMPLES,config.CORR_SAMPLES,config.GRID_SIZE,config.GRID_SIZE), 
            device=device
        )
        stats['fixation_map'] = torch.zeros((crange, crange))
        stats['avg_reco'] = 0
        stats['threshs'] = 0
        stats['max_lat'] = 0
        stats['avg_input'] = torch.zeros((1,2,cropsize,cropsize), device=device)
        stats['reco_strength'] = 1
    stats['slow_beta'] = 1e-4
    stats['fast_beta'] = 1e-3
    
    # refresh these only if I'm not just plotting stuff
    if not config.PRINT and not config.RECOPRINT:
        stats['map_tracker'] = torch.zeros(config.N_BATCHES//config.MAP_SAMPLS+1, config.GRID_SIZE, config.GRID_SIZE)
        stats['affinity_tracker'] = torch.zeros(config.N_BATCHES+1)
        stats['reco_tracker'] = torch.zeros(config.N_BATCHES+1)
        stats['lat_tracker'] = torch.zeros((config.N_BATCHES+1,config.GRID_SIZE,config.GRID_SIZE))
    return stats

# perform a reconstruction step
def reco_step(model, compression_kernel, stats, reco_target=1.15):
    
    lat = model.last_lat
    csize = compression_kernel.shape[-1]
    border_trim = round(0.1*lat.shape[-1]) + 1
    # compute the reconstruction by feeding back and forth through the sparse projection
    pad = config.COMPRESSION*csize
    indices = torch.arange(lat.shape[-1]//config.COMPRESSION)*config.COMPRESSION
    compressed = lat[0,0,indices[:,None],indices[None]][None,None]
    reco = F.conv_transpose2d(compressed, compression_kernel, stride=config.COMPRESSION, padding=csize//2)
    gap = lat.shape[-1] - reco.shape[-1]
    reco = F.pad(reco, (0,gap,0,gap))
    #reco *= stats['reco_strength']
    reco *= config.COMPRESSION**2 * 1.4

    with torch.no_grad():
        lat_reco = model(
            reco, 
            reco_flag=True,
            learning_flag=False 
        )
        
    
    #lat_reco_max = lat_reco.max().float()
    #if lat_reco_max > 0.1:
    #    stats['reco_strength'] -= (lat_reco_max - reco_target) * 1e-2
    # trimming away the borders before computing the cosine similarity to avoid border effects
    lat_reco_trimmed = lat_reco[:,:,border_trim:-border_trim,border_trim:-border_trim].float()
    lat_reco_trimmed_norm = torch.sqrt((lat_reco_trimmed**2).sum()) + 1e-7
    lat_reco_trimmed_unit = lat_reco_trimmed / lat_reco_trimmed_norm
    lat_trimmed = lat[:,:,border_trim:-border_trim,border_trim:-border_trim].float()
    lat_trimmed_norm = torch.sqrt((lat_trimmed**2).sum()) + 1e-7
    lat_trimmed_unit = lat_trimmed / lat_trimmed_norm
    diff = (lat_trimmed_unit * lat_reco_trimmed_unit).sum()
    #base = lat_trimmed_unit.mean() * np.sqrt(model.sheet_units**2)
    #diff = torch.relu(diff - base) / (1 - base)
        
    return diff, lat_reco_trimmed, lat_trimmed, reco, compressed

# plot the steps of the reconstruction
def plot_reco_steps(reco, sample, lat, lat_reco, compressed, diff):
    
    print('reco, sample, lat, lat_reco')
    plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    plt.imshow(reco[0,0].cpu().float())
    plt.subplot(2,2,2)
    plt.imshow(sample[0,0].cpu().float())
    plt.subplot(2,2,3)
    plt.imshow(lat[0,0].cpu().float())
    plt.subplot(2,2,4)
    plt.imshow(((lat_reco).abs())[0,0].cpu().float())
    plt.show()
    plt.imshow(compressed[0,0].cpu().float())
    plt.show()
    print('reco max: ', reco.max())
    print('reco: ', lat_reco.max(), 'lat: ', lat.max())
    print('score: ', diff)
    
# collect the stats for this iteration
def collect_stats(idx, stats, model, sample, compression_kernel):

    rfs_det = model.get_rfs()
    # if it is the right time and we are not in printing mode, get a snapshot of the map
    if idx%config.MAP_SAMPLS==0 and not config.PRINT:
        stats['map_tracker'][idx//config.MAP_SAMPLS] = get_orientations(model)[0]
        
    # compute the first component of the metric
    x_tiles = model.x_tiles
    raw_aff = model.aff_cache
    lat = model.last_lat
    stats['lat_tracker'][idx] = lat[0,0].cpu()
    aff_flat = raw_aff.detach().view(-1).float()
    x_norm_squared = (x_tiles**2).sum(1, keepdim=True)
    #x_norm = torch.sqrt(x_norm_squared)
    #x_unit = x_tiles / (x_norm + 1e-7)
    #x_base = (x_unit * model.rf_envelope).sum(1).view(-1)
    #x_base = x_unit.mean() * np.sqrt(x_tiles.shape[1])
    rfs_norm_squared = (rfs_det**2).sum(1, keepdim=True)
    norms = torch.sqrt(rfs_norm_squared * x_norm_squared).view(-1)
    cos_sims = aff_flat  / (norms  + 1e-7)
    #cos_sims = (cos_sims - x_base) / (1 - x_base)
    rf_affinity = (cos_sims * lat.view(-1)) / (lat.sum() + 1e-7)
    rf_affinity = rf_affinity.sum()
            
    incorporation = lat.mean() / model.homeo_target
    beta = incorporation * stats['fast_beta']
    stats['rf_affinity'] = (1-beta)*stats['rf_affinity'] + beta*rf_affinity
    stats['affinity_tracker'][idx] = stats['rf_affinity']
    
    # pick some samples of global correlations, used too see how well the map decorrelates
    gs = config.GRID_SIZE
    cs = config.CORR_SAMPLES
    factors = lat[0,0,gs//2-cs//2:gs//2+cs//2+1,gs//2-cs//2:gs//2+cs//2+1]
    factors = factors.view(cs,cs,1,1)
    stats['global_corr'] = stats['global_corr']*(1-stats['slow_beta']) + lat*factors*stats['slow_beta']
    stats['avg_input'] = stats['avg_input']*(1-stats['slow_beta']) + sample*stats['slow_beta']
    stats['avg_acts'] = (1-stats['fast_beta'])*stats['avg_acts'] + stats['fast_beta']*model.lat_mean
    avg_acts_mean = stats['avg_acts'].mean()
    stats['threshs'] = model.adathresh.mean()
    
    # if flag is on, perform a step of the reconstruction and store the metric
    if config.RECO or config.RECOPRINT:
        # getting the reconstruction score
        diff, lat_reco_trimmed, lat_trimmed, reco, compressed = reco_step(model, compression_kernel, stats)
        # only store the metric if it is above zero (activation is present)
        #if diff:
        stats['avg_reco'] = (1-beta)*stats['avg_reco'] + beta*diff
        stats['reco_tracker'][idx] = stats['avg_reco']
        # plotting the different phases of the reconstruction
        if config.RECOPRINT and not config.PRINT:
            plot_reco_steps(reco, sample, lat_trimmed, lat_reco_trimmed, compressed, diff)
            
    # store the max of the activations to assess whether the automatic strength tuning works
    curr_max_lat = lat.float().max()
    if curr_max_lat > 0.1:
        stats['max_lat'] = (1-stats['fast_beta'])*stats['max_lat'] + stats['fast_beta']*curr_max_lat

# update the progress bar with the new values
def update_progress_bar(progress_bar, stats):
    progress_bar.set_description(\
                "affinity: {:.3f} reco: {:.3f} ensemble: {:.3f} avg act: {:.3f} max: {:.3f} threshs: {:.3f} LR: {:.4f}".format\
            (
                stats['rf_affinity'], 
                stats['avg_reco'], 
                stats['rf_affinity']*stats['avg_reco'],
                stats['avg_acts'].mean(), 
                stats['max_lat'], 
                stats['threshs'], 
                stats['lr']
            ))