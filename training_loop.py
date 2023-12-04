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
import nn_template

def run(X, model=None, stats=None, bar=True, eval_at_end=False):
        
    if stats==None:
        stats = {}
        
    device = X.device
    # randomly permuting the blocks of single toy sequences
    channels = X.shape[1]
    X = X.type(config.DTYPE)
    xshape = X.shape
    input_size = xshape[-1]
    X = X.view(-1, config.FRAMES_PER_TOY, channels, input_size, input_size)
    X = X[torch.randperm(X.shape[0])]
    X = X.view(xshape)
    
    # defining the model
    if not model:
        model = new_model(X)
    
    # defining the compression kernel
    cstd = (config.COMPRESSION-1) + 0.1
    csize = oddenise(round(cstd*model.cutoff))
    compression_kernel = get_gaussian(csize,cstd) * get_circle(csize,csize/2)
    compression_kernel /= compression_kernel.sum()
    compression_kernel = compression_kernel.cuda().type(config.DTYPE)
    
    sparse_masks = get_weight_masks(config.SPARSITY, model.lat_weights.shape).cuda().type(config.DTYPE)
    rf_sparse_masks = get_weight_masks(config.RF_SPARSITY, model.rfs.shape).cuda().type(config.DTYPE)
    
        
    # defining the variables for the saccades
    input_pad = round(config.RF_STD*model.cutoff)//2
    cropsize = config.CROPSIZE + input_pad*2
    crange = input_size - cropsize
    frame_idx = 0
    cx, cy = crange//2, crange//2
                
    init_stats_dict(stats, device, crange, cropsize, config.EVALUATE and not eval_at_end)
    
    # defining the optimiser, the scheduler and the progress bar
    lr = config.LR
    lr_decay = (1-1/config.N_BATCHES*np.log(config.TARGET_LR_DEC))
    optimiser = torch.optim.SGD([model.rfs, model.lat_weights], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=lr_decay)
    progress_bar = enumerate(range(config.N_BATCHES+1))
    if bar:
        progress_bar = tqdm(progress_bar, position=0, leave=True)

    # main traning loop 
    for _,idx in progress_bar:
            
        # if it is time to change toy, select a new one and transform it by a random angle
        if idx%config.FRAMES_PER_TOY==0:
            toy_idx = random.randint(0, X.shape[0]-1)
            frame_idx = 0
            while X[toy_idx,0].mean() < 0.:
                toy_idx = random.randint(0, X.shape[0]//config.FRAMES_PER_TOY)
            X_toy = X[toy_idx*config.FRAMES_PER_TOY:(toy_idx+1)*config.FRAMES_PER_TOY]     
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
            learning_flag=config.LEARNING and eval_at_end
        )

        if config.LEARNING and eval_at_end:
            # compute the loss value and perform one step of traning
            loss = cosine_loss(model)
            if loss!=0:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
        stats['lr'] = lr
                    
        stats['lat_tracker'][idx] = model.last_lat[0,0].cpu()
        stats['x_tracker'][idx] = sample[:,:,input_pad:-input_pad,input_pad:-input_pad]
        stats['sqr_tracker'][idx] = model.sqr_lat[0,0].cpu() / config.ITERS
        if idx%config.STATS_FREQ==0:
            collect_stats(idx, stats, model, sample, compression_kernel, sparse_masks, rf_sparse_masks, not eval_at_end)
        
        if config.PRINT or config.RECOPRINT:
            print('abort')
            break
            
        if bar:
            update_progress_bar(progress_bar, stats)
            progress_bar.update()
      
    #train_NN(stats)
    if eval_at_end and False:
        prev_batch = config.N_BATCHES
        prev_stat = config.STATS_FREQ
        config.STATS_FREQ = 1
        config.N_BATCHES = config.EVAL_BATCHES
        run(X, model, stats, bar=False)
        config.STATS_FREQ = prev_stat
        config.N_BATCHES = prev_batch
        stats['avg_reco'] = (stats['reco_tracker']).mean()
        stats['rf_affinity'] = (stats['affinity_tracker']).mean()
        stats['avg_noise'] = (stats['noise_tracker']).mean(0)
        stats['avg_corr_noise'] = (stats['corr_noise_tracker']).mean(0)
        stats['avg_fixed_noise'] = (stats['fixed_noise_tracker']).mean(0)
        stats['avg_sparsity'] = (stats['sparsity_tracker']).mean(0)
        stats['avg_rf_sparsity'] = (stats['rf_sparsity_tracker']).mean(0)
        stats['avg_mixed_case'] = (stats['mixed_case_tracker']).mean(0)
        sig_pow = (stats['sqr_tracker']).mean()
        stats['stn'] = torch.tensor([round(10*math.log10(sig_pow/noise**2)) for noise in config.NOISE])
        print('final stats, affinity: {:.3f} reco {:.3f}'.format(
            stats['rf_affinity'],
            stats['avg_reco']
            )
        )
        print('avg_fixed noise: ', stats['avg_fixed_noise'])
        print('avg noise: ', stats['avg_noise'])
        print('avg corr noise: ', stats['avg_corr_noise'])
        print('signal to noise: ', stats['stn'])
        print('avg sparsity: ', stats['avg_sparsity'])
        print('avg RF sparsity: ', stats['avg_rf_sparsity'])
        print('mixed case: ', stats['avg_mixed_case'])
            
    return model, stats


def new_model(X):
    return CorticalMap(
        X.device,
        config.GRID_SIZE,
        config.RF_STD,
        X.shape[1],
        config.EXPANSION,
        config.EXC_STD,
        config.DILATION,
        config.ITERS,
        config.V1_SATURATION,
        config.TARGET_ACT,
        config.HOMEO_TIMESCALE,
        config.DTYPE
    ).to(X.device)

def train_NN(stats, debug=False):
    
    samples = 10000
    X = stats['lat_tracker'][-samples:,None].cuda()
    Y = stats['x_tracker'][-samples:].cuda()
    stats['nn_loss_tracker'] = torch.zeros(config.NN_EPOCHS)
    
    split = round(samples*0.8)
    X_train = X[:split]
    Y_train = Y[:split]
    X_test = X[split:]
    Y_test = Y[split:]
    
    input_size = X.shape[-1]
    output_size = Y.shape[-1]
    w = 5
    network_hidden = [
        ('flatten', 1, input_size**2),
        #('unflatten', 1, size*2),
        #('conv2d', 64, 41),
        #('pool2d', 64, 2),
        #('conv2d', 64, w),
        #('conv2d', 32, w),
        #('conv2d', 16, w),
        #('conv2d', 8, w),
        #('conv2d', 16, w),
        #('conv2d', 2, w),
        ('dense', 2*output_size**2, input_size**2),
        #('dense', 2*size**2, 2*size**2),
        ('unflatten', 2, output_size)
    ]
    
    
    stats['network'] = nn_template.Network(network_hidden, device=X.device , bias=config.BIAS)
    params_list = [list(stats['network'].layers[l].parameters()) for l in range(len(network_hidden))]
    params_list = sum(params_list, [])
    optim = torch.optim.Adam(params_list, lr=1e-4)
    loss_avg = 0
    beta = 0.99
    
    progress_bar = tqdm(range(config.NN_EPOCHS))
    for b in progress_bar:
        
        batch_idx = torch.randint(0, split, (config.BSIZE,))
        x_b = X_train[batch_idx]
        y_b = Y_train[batch_idx]
                
        reco = torch.relu(stats['network'](x_b, debug=debug))
        loss = ((y_b - reco)**2).sum([1,2,3]).mean() + (stats['network'].layers[1].weight.abs()).sum()*1e-3
        stats['nn_loss_tracker'][b] = int(loss.detach())
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        loss_avg = beta*loss_avg + (1-beta)*stats['nn_loss_tracker'][b]
        progress_bar.set_description("reco loss: " + str(int(loss_avg)))
        
    test_reco = torch.relu(stats['network'](X_test, debug=debug))
    test_loss = ((Y_test - test_reco)**2).sum([1,2,3]).mean()
    stats['test_loss'] = test_loss.detach()
        
