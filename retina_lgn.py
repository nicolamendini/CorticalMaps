# Author: Nicola Mendini
# Code used to simulate the Retinal and LGN processed

import torch
from maps_helper import *

# function to process the data by simulating retina and lgn
def retina_lgn(
    X,
    ret_log_std,
    ret_strength,
    lgn_strength,
    lgn_log_std,
    hard_gc,
    gcstd,
    gc_scaler,
    gc_eps,
    same_mean
):
    
    # simulating the retinal processes
    ret_log_size = ret_log_std*13
    ret_log_size = ret_log_size+1 if ret_log_size%2==0 else ret_log_size
    log = (get_log(ret_log_size, ret_log_std)  * get_circle(ret_log_size, ret_log_size)) * ret_strength
    log = log.to(X.device).type(torch.float16) 
    log /= torch.sqrt((log**2).sum())
    X = F.pad(X,(ret_log_size//2,ret_log_size//2,ret_log_size//2,ret_log_size//2), mode='reflect')
    X = torch.cat([F.conv2d(X, -log), F.conv2d(X, log)], dim=1)
    X = torch.relu(X)

    # applying another stage of on/off filtering if required
    if lgn_strength:
        lgn_log_size = lgn_log_std*13
        lgn_log_size = lgn_log_size+1 if lgn_log_size%2==0 else lgn_log_size
        log = (get_log(lgn_log_size, lgn_log_std) * get_circle(lgn_log_size, lgn_log_size/2)) * lgn_strength
        log = log.to(X.device).type(torch.float16)
        log /= torch.sqrt((log**2).sum())
        X = F.pad(X,(lgn_log_size//2,lgn_log_size//2,lgn_log_size//2,lgn_log_size//2), mode='reflect')
        X = torch.cat(
            [
                F.conv2d(X[:,0][:,None], -log), 
                F.conv2d(X[:,1][:,None], log)
            ], dim=1
        )
        X = torch.relu(X)

    # saturating the activations
    X = torch.tanh(X)
    X_orig = X+0
    X_mask = X>0

    # LGN stages. Applying gain control as implemented in stevens 2008 GCAL
    if hard_gc:
        gcsize = round(gcstd*5)
        gcsize = gcsize+1 if gcsize%2==0 else gcsize
        gc_kernel = (get_radial_cos(gcsize, gcsize)**2 * get_circle(gcsize, gcsize/2))
        gc_kernel = gc_kernel.type(torch.float16).to(X.device) / gc_kernel.sum()
        for i in range(X.shape[0]):
            localmax = F.unfold(X[i:i+1], gcsize, padding=gcsize//2) 
            localmax = localmax.view(1, 2, gcsize**2, -1) * gc_kernel.view(1,1,-1,1)  
            localmax = localmax.sum(2, keepdim=True).view(1,2,X.shape[-1],X.shape[-1])
            X[i:i+1] /= (localmax + gc_eps) / gc_scaler



    X_cg = X + 0

    # if on, ensures that each input units has the same average activation over time
    if same_mean:
        X_mean = X.mean(0, keepdim=True)
        X = X / (X_mean + 1e-10) * X_mean.mean([-1,-2], keepdim=True)
        X_mean = X.mean(0, keepdim=True)
        print(X_mean[0,0].min(), X_mean[0,0].max(),X_mean[0,1].min(), X_mean[0,1].max())

    #saturating again
    X = torch.tanh(X)
    
    return X, X_orig, X_mask