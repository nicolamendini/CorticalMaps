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
from fast_gcal import *

def run(X, model=None, stats=None, bar=True):
    
    lr = config.LR
    channels = X.shape[1]
    cstd = 0.8*config.COMPRESSION*5
    csize = round(cstd)
    csize = csize+1 if csize%2==0 else csize
    border_trim = csize//2
    compression_kernel = get_radial_cos(csize,cstd)**2 * get_circle(csize,cstd/2)
    compression_kernel /= compression_kernel.sum()
    compression_kernel = compression_kernel.cuda()
    device = X.device
    compression_kernel = compression_kernel.type(config.DTYPE)
    X = X.type(config.DTYPE)
                
    if not model:
        model = CorticalMap(
            device,
            config.GRID_SIZE,
            config.RF_SIZE,
            channels,
            config.EXPANSION,
            config.DILATION,
            config.ITERS,
            config.STRENGTH,
            config.TARGET_ACT,
            config.HOMEO_TIMESCALE,
            config.EXC_SCALE,
            config.INH_SCALE,
            config.MAX_INH_FAC,
            config.DTYPE
        ).to(device)
    
    # if theres no stats dictionary
    if not stats:
        stats = {}
        stats['rf_affinity'] = 0
        stats['reco_rf_affinity'] = 0
        stats['avg_acts'] = torch.zeros((1,1,config.GRID_SIZE,config.GRID_SIZE), device=device) + config.TARGET_ACT
        stats['global_corr'] = torch.zeros(
            (config.CORR_SAMPLES,config.CORR_SAMPLES,config.GRID_SIZE,config.GRID_SIZE), 
            device=device
        )
        stats['last_lat'] = torch.zeros(1,1,config.GRID_SIZE,config.GRID_SIZE).to(device)
        stats['avg_reco'] = 0
        stats['last_sample'] = 0
        stats['threshs'] = 0
        stats['max_lat'] = 0
    stats['global_corr_beta'] = 1e-4
    stats['avg_act_beta'] = 1e-3
    stats['rf_beta'] = 1e-3
    stats['strength_beta'] = 3e-3
    
    # refresh these only if I'm not just plotting stuff
    if not config.PRINT and not config.RECOPRINT:
        stats['map_tracker'] = torch.zeros(config.N_BATCHES//config.MAP_SAMPLS+1, config.GRID_SIZE, config.GRID_SIZE)
        stats['affinity_tracker'] = torch.zeros(config.N_BATCHES+1)
        stats['reco_tracker'] = torch.zeros(config.N_BATCHES+1)
    
    optimiser = torch.optim.SGD([model.rfs, model.lat_weights], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=config.LR_DECAY)
    
    progress_bar = enumerate(range(config.N_BATCHES+1))
    if bar:
        progress_bar = tqdm(progress_bar, position=0, leave=True)
    else:
        print('run starting, scale is: ', config.SCALE)

    for _,idx in progress_bar:
        
        if idx%config.MAP_SAMPLS==0 and not config.PRINT:
            stats['map_tracker'][idx//config.MAP_SAMPLS] = get_orientations(
                model.get_rfs(), config.RF_SIZE, config.GRID_SIZE)[0]

        rand_idx = random.randint(0, X.shape[0]-1)
        sample = X[rand_idx:rand_idx+1]
        
        input_pad = config.RF_SIZE//2
        sample = TF.rotate(sample, random.randint(0,360), interpolation=TF.InterpolationMode.BILINEAR)

        if random.random() > 0.5:
            sample = sample.flip(-1)

        new_w = sample.shape[-1]
        cropsize = config.CROPSIZE + input_pad*2
        crange = new_w-cropsize
        cx = random.randint(0,crange)
        cy = random.randint(0,crange)

        sample = sample[:,:,cx:cx+cropsize,cy:cy+cropsize]
        
        if random.random() > 0.5:
            sample = sample.flip(-1)
                      
        target_size = config.GRID_SIZE
        # tha padding should not go below an expansion of 1
        target_size += input_pad*2*max(round(config.EXPANSION),1)
        sample = F.interpolate(sample, target_size, mode='bilinear')
        
        if config.PRINT:
            plt.imshow(sample[0,0].cpu())
            plt.show()
                                    
        raw_aff, lat, lat_correlations, x_tiles = model(
            sample, 
            reco_flag=False, 
            learning_flag=config.LEARNING,
            print_flag=config.PRINT
        )
        
        stats['avg_acts'] = (1-stats['avg_act_beta'])*stats['avg_acts'] + stats['avg_act_beta']*model.lat_mean
        avg_acts_mean = stats['avg_acts'].mean()
        stats['threshs'] = model.adathresh.mean()

        # do not train and do not take stats if lat is 0
        stats_flag = idx%config.STATS_FREQ==0 and config.STATS_FREQ!=-1

        if config.LEARNING:
        
            loss = cosine_loss(raw_aff, lat, lat_correlations, model)

            if loss!=0:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        if config.RECO or config.RECOPRINT:
            
            # only do the full thing is you are printing or if you have to collect the stat
            if config.RECOPRINT or (config.RECO and not config.RECOPRINT):

                pad = config.COMPRESSION*csize
                indices = torch.arange(lat.shape[-1]//config.COMPRESSION)*config.COMPRESSION
                compressed = lat[0,0,indices[:,None],indices[None]][None,None]
                reco = F.conv_transpose2d(compressed, compression_kernel, stride=config.COMPRESSION, padding=csize//2)
                gap = lat.shape[-1] - reco.shape[-1]
                reco = F.pad(reco, (0,gap,0,gap))
                reco *= config.COMPRESSION**2 
                            
                with torch.no_grad():
                    _,lat_reco,_,_ = model(
                        reco, 
                        reco_flag=True,
                        learning_flag=False, 
                        print_flag=config.PRINT
                    )

                # trimming away the borders before computing the cosine similarity to avoid border effects
                lat_reco_trimmed = lat_reco[:,:,border_trim:-border_trim,border_trim:-border_trim].float()
                lat_trimmed = lat[:,:,border_trim:-border_trim,border_trim:-border_trim].float()
                norm = torch.sqrt((lat_reco_trimmed**2).sum() * (lat_trimmed**2).sum())
                diff = (lat_trimmed * lat_reco_trimmed).sum() / (norm + 1e-5)        
                if diff:
                    reco_max_lat = lat_reco.max()
                    stats['avg_reco'] = (1-stats['rf_beta'])*stats['avg_reco'] + stats['rf_beta']*diff
                stats['reco_tracker'][idx] = stats['avg_reco']

                if config.RECOPRINT:
                    
                    print('reco, sample, lat, lat_reco')
                    plt.figure(figsize=(10,10))
                    plt.subplot(2,2,1)
                    plt.imshow(reco[0,0].cpu())
                    plt.subplot(2,2,2)
                    plt.imshow(sample[0,0].cpu())
                    plt.subplot(2,2,3)
                    plt.imshow(lat[0,0].cpu())
                    plt.subplot(2,2,4)
                    plt.imshow(((lat_reco-lat).abs())[0,0].cpu())
                    plt.show()
                    plt.imshow(compressed[0,0].cpu())
                    plt.show()
                    print('reco max: ', reco.max())
                    print('reco: ', lat_reco.max(), 'lat: ', lat.max())
                    print('score: ', diff)
                    break
                    
        if config.PRINT:
            print('abort')
            break
                    
        if stats_flag:
            stats['last_sample'] = sample
            stats['last_lat'] = lat
            
            rfs_det = model.get_rfs()
            aff_flat = raw_aff.detach().view(-1).float()
            norms = torch.sqrt((rfs_det**2).sum(1) * (x_tiles**2).sum(2)).view(-1)
            cos_sims = aff_flat * lat.view(-1) / (norms * lat.sum() + 1e-5)
            #reco_cos_sims = aff_flat * lat_reco.view(-1) / (norms * lat_reco.sum() + 1e-5)
            rf_affinity = cos_sims.sum()
            #reco_rf_affinity = reco_cos_sims.sum()
            stats['rf_affinity'] = (1-stats['rf_beta'])*stats['rf_affinity'] + stats['rf_beta']*rf_affinity
            #stats['reco_rf_affinity'] = (1-stats['rf_beta'])*stats['reco_rf_affinity'] + \
            #    stats['rf_beta']*reco_rf_affinity
            stats['affinity_tracker'][idx] = stats['rf_affinity']
            
            gs = config.GRID_SIZE
            cs = config.CORR_SAMPLES
            factors = lat[0,0,gs//2-cs//2:gs//2+cs//2+1,gs//2-cs//2:gs//2+cs//2+1]
            factors = factors.view(cs,cs,1,1)
            stats['global_corr'] = stats['global_corr']*(1-stats['global_corr_beta']) + \
                lat*factors*stats['global_corr_beta']
        if bar:
            progress_bar.set_description(\
                "affinity: {:.3f} reco cos: {:.3f} ensemble: {:.3f} avg act: {:.3f} max act: {:.3f} threshs: {:.3f} LR: {:.3f}".format\
            (
                stats['rf_affinity'], 
                stats['avg_reco'], 
                stats['rf_affinity']*stats['avg_reco'],
                avg_acts_mean, 
                stats['max_lat'], 
                stats['threshs'], 
                lr
            ))
            progress_bar.update()
        
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        
        curr_max_lat = lat.float().max()
        if curr_max_lat > 0.1:
            stats['max_lat'] = (1-stats['strength_beta'])*stats['max_lat'] + stats['strength_beta']*curr_max_lat
            
    return model, stats