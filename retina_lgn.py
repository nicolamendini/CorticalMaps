# Author: Nicola Mendini
# Code used to simulate the Retinal and LGN processed

import torch
from maps_helper import *

# function to process the data by simulating retina and lgn
def retina_lgn(
    X,
    ret_log_std,
    lgn_log_std,
    hard_gc,
    gcstd,
    gc_eps,
    saturation,
    lgn_iters,
    target_max=1,
    eps_thresh=1e-3
):
    
    # simulating the retinal processes
    ret_log_size = round(ret_log_std*13)
    ret_log_size = oddenise(ret_log_size)
    log = (get_log(ret_log_size, ret_log_std)  * get_circle(ret_log_size, ret_log_size)) 
    log = log.to(X.device).type(torch.float16)
    X = F.pad(X,(ret_log_size//2,ret_log_size//2,ret_log_size//2,ret_log_size//2), mode='reflect')
    X = torch.cat([F.conv2d(X, -log), F.conv2d(X, log)], dim=1)
    X = torch.relu(X-eps_thresh)
    
    X_orig = X[:]
    
    # if on, ensures that each input units has the same average activation over time
    X_mean = X.mean([0,2,3], keepdim=True)
    X = X / (X_mean + 1e-10)
    X_mean = X.mean([0,2,3])
        
    
    if lgn_iters:
        # applying another stage of on/off filtering if required
        lgn_log_size = round(lgn_log_std*13)
        lgn_log_size = oddenise(lgn_log_size)
        log = (get_log(lgn_log_size, lgn_log_std) * get_circle(lgn_log_size, lgn_log_size/2)) 
        log = log.to(X.device).type(torch.float16)
        for i in range(lgn_iters):
            X = F.pad(X,(lgn_log_size//2,lgn_log_size//2,lgn_log_size//2,lgn_log_size//2), mode='reflect')
            X = torch.cat(
                [
                    F.conv2d(X[:,0][:,None], -log), 
                    F.conv2d(X[:,1][:,None], -log)
                ], dim=1
            )
            
            X = torch.relu(X)
            X_mean = X.mean([0,2,3], keepdim=True)
            X = X / (X_mean + 1e-10)
            X_mean = X.mean([0,2,3])
            
    X_mask = X>0

    # LGN stages. Applying gain control as implemented in stevens 2008 GCAL
    if hard_gc:
        gcsize = round(gcstd*5)
        gcsize = oddenise(gcsize)
        gc_kernel = (get_radial_cos(gcsize, gcsize)**2 * get_circle(gcsize, gcsize/2))
        gc_kernel = gc_kernel.type(torch.float16).to(X.device) / gc_kernel.sum()
        
        for i in range(X.shape[0]):
            localmax = F.unfold(X[i:i+1], gcsize, padding=gcsize//2) 
            localmax = localmax.view(1, 2, gcsize**2, -1) * gc_kernel.view(1,1,-1,1)  
            localmax = localmax.sum(2, keepdim=True).view(1,2,X.shape[-1],X.shape[-1])
            X[i:i+1] /= (localmax + gc_eps) 
        
        X = torch.tanh(X*saturation)
        
        # if on, ensures that each input units has the same average activation over time
        X_mean = X.mean([0,2,3], keepdim=True)
        X = X / (X_mean + 1e-10)
        X_mean = X.mean([0,2,3])
        X = X / X.max() * target_max
        print(X_mean[0], X_mean[1], X.max(), X.min())
        
    
    return X, X_orig, X_mask