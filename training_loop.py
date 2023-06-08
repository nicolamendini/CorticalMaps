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
from cortical_map import *
from training_helper import *

def run(X, model=None, stats=None, bar=True):
    
    # randomly permuting the blocks of single toy sequences
    channels = X.shape[2]
    X = X.type(config.DTYPE)
    input_size = X.shape[-1]
    X = X.view(-1, config.FRAMES_PER_TOY, channels, input_size, input_size)
    X = X[torch.randperm(X.shape[0])]
    X = X.view(-1, 1, channels, input_size, input_size)
    device = X.device
    
    # defining the model
    if not model:
        model = CorticalMap(
        device,
        config.GRID_SIZE,
        config.RF_STD,
        channels,
        config.EXPANSION,
        config.EXC_STD,
        config.DILATION,
        config.ITERS,
        config.V1_SATURATION,
        config.TARGET_ACT,
        config.HOMEO_TIMESCALE,
        config.DTYPE
        ).to(device)
    
    # defining the compression kernel
    cstd = 0.8*config.COMPRESSION
    csize = oddenise(round(cstd*model.cutoff))
    compression_kernel = get_gaussian(csize,cstd) * get_circle(csize,csize/2)
    compression_kernel /= compression_kernel.sum()
    compression_kernel = compression_kernel.cuda().type(config.DTYPE)
    
    # defining the variables for the saccades
    input_pad = round(config.RF_STD*model.cutoff)//2
    cropsize = config.CROPSIZE + input_pad*2
    crange = input_size - cropsize
    frame_idx = 0
    cx, cy = crange//2, crange//2
                
    stats = init_stats_dict(stats, device, crange, cropsize)
    
    # defining the optimiser, the scheduler and the progress bar
    lr = config.LR
    lr_decay = (1-1/config.N_BATCHES*np.log(config.TARGET_LR_DEC))
    optimiser = torch.optim.SGD([model.rfs, model.lat_weights], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=lr_decay)
    progress_bar = enumerate(range(config.N_BATCHES+1))
    if bar:
        progress_bar = tqdm(progress_bar, position=0, leave=True)
    else:
        print('run starting, scale is: ', config.SCALE)

    # main traning loop 
    for _,idx in progress_bar:
            
        # if it is time to change toy, select a new one and transform it by a random angle
        if idx%config.FRAMES_PER_TOY==0:
            toy_idx = random.randint(0, X.shape[0]-1)
            frame_idx = 0
            while X[toy_idx,0].mean() < 0.:
                toy_idx = random.randint(0, X.shape[0]-1)
            X_toy = X[toy_idx]     
            rand_rot_angle = random.randint(0,360)
            X_toy = TF.rotate(X_toy, rand_rot_angle, interpolation=TF.InterpolationMode.BILINEAR)
        # otherwise, just pick another frame of that same toy
        else:
            frame_idx += 1

        # perform a levy step, where the max step is the size of the image itself
        cx, cy = levy_step(crange, cx, cy, min_step_size=config.MIN_STEP, long_step_p=config.LONG_STEP_P)
        sample = X_toy[frame_idx:frame_idx+1,:,cx:cx+cropsize,cy:cy+cropsize]
        stats['fixation_map'][cx, cy] += 1
                
        if config.PRINT:
            plt.imshow(sample[0,0].cpu().float())
            plt.show()
        
        # feed the sample into the model
        model(
            sample, 
            reco_flag=False, 
            learning_flag=config.LEARNING
        )

        if config.LEARNING:
            # compute the loss value and perform one step of traning
            loss = cosine_loss(model)
            if loss!=0:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        stats['lr'] = lr
                    
        collect_stats(idx, stats, model, sample, compression_kernel)
        
        if config.PRINT or config.RECOPRINT:
            print('abort')
            break
            
        if bar:
            update_progress_bar(progress_bar, stats)
            progress_bar.update()
            
    return model, stats