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
    eps_thresh=1e-2
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
    
    # ensures that each input units has the same average activation over time
    #X_mean = X.mean([0,2,3], keepdim=True)
    #X = X / (X_mean + 1e-10)
    #X_mean = X.mean([0,2,3])
        
    
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
        gcsize = round(gcstd*6)
        gcsize = oddenise(gcsize)
        #gc_kernel = (get_radial_cos(gcsize, gcsize)**2 * get_circle(gcsize, gcsize/2))
        gc_kernel = get_gaussian(gcsize, gcstd) * get_circle(gcsize, gcsize/2)
        gc_kernel = gc_kernel.type(torch.float16).to(X.device) / gc_kernel.sum()
        
        for i in range(X.shape[0]):
            localmax = F.unfold(X[i:i+1], gcsize, padding=gcsize//2) 
            localmax = localmax.view(1, 2, gcsize**2, -1) * gc_kernel.view(1,1,-1,1)  
            localmax = localmax.sum(2, keepdim=True).view(1,2,X.shape[-1],X.shape[-1])
            X[i:i+1] /= (localmax + gc_eps) 
        
        X = torch.tanh(X*saturation)
        
        # if on, ensures that each input units has the same average activation over time
        #X_mean = X.mean([0,2,3], keepdim=True)
        #X = X / (X_mean + 1e-10)
        #X_mean = X.mean([0,2,3])
        #X = X / X.max() * target_max
        #print(X_mean[0], X_mean[1], X.max(), X.min())
        
    
    return X, X_orig, X_mask


def plot_input_stats(X, X_orig, X_mask, size):

    plt.figure(figsize=(size,size))
    plt.subplots_adjust(
                    hspace=0.3,
                    wspace=0.4
    )
    s = random.randint(0,X.shape[0]-1)
    plt.subplot(3,3,1)
    plt.gca().set_title('original off')
    plt.imshow(X_orig[s,0].cpu())
    plt.subplot(3,3,2)
    plt.gca().set_title('original on')
    plt.imshow(X_orig[s,1].cpu())
    plt.subplot(3,3,3)
    plt.gca().set_title('cg off')
    plt.imshow(X[s,0].cpu())
    plt.subplot(3,3,4)
    plt.gca().set_title('cg on')
    plt.imshow((X[s,1]).cpu())
    plt.subplot(3,3,5)
    plt.gca().set_title('cg off hist')
    #plt.yscale('log')

    #plt.yscale('log')
    x_plot_off = X[:,0][X_mask[:,0]].cpu().numpy().flatten()
    x_off_mean = x_plot_off.mean()
    plt.hist(x_plot_off, bins=100)
    plt.axvline(x_off_mean, color='k', linestyle='dashed', linewidth=1)
    plt.subplot(3,3,6)
    plt.gca().set_title('cg on hist')
    #plt.yscale('log')

    #plt.yscale('log')
    x_plot_on = X[:,1][X_mask[:,1]].cpu().numpy().flatten()
    x_on_mean = x_plot_on.mean()
    plt.hist(x_plot_on, bins=100)
    plt.axvline(x_on_mean, color='k', linestyle='dashed', linewidth=1)

    # Measure how similar the data is between before and after contrast gain
    cos_sim = 0
    for i in range(X.shape[0]):
        cos_sim += cosine_sim(X_orig[i][X_mask[i]], X[i][X_mask[i]])
    cos_sim /= X.shape[0]

    kurt_on = get_fisher_kurtosis(x_plot_on)
    kurt_off = get_fisher_kurtosis(x_plot_off)
    general_std = (x_plot_on.std() + x_plot_off.std())/2
    rmse = 1-torch.sqrt((X_orig - X)**2).mean()

    fou_crop = 40
    match_std = 0.8
    fou_on = torch.fft.fft2(X).abs().mean([0,1]).cpu()
    fou_on = torch.fft.fftshift(fou_on)
    fou_on_plot = fou_on.sum(0)
    fou_on_plot -= fou_on_plot.mean()*1.9
    fou_on_plot /= fou_on_plot.max()
    fou_on_plot = fou_on_plot[fou_crop:-fou_crop]
    match = get_gaussian(fou_on.shape[-1]-fou_crop*2, match_std)
    match = match.sum([0,1,3])
    match /= match.max()

    plt.subplot(3,3,7)
    plt.gca().set_title('first sample of cg dataset')
    plt.imshow(X[0,0].cpu())
    plt.subplot(3,3,8)
    plt.gca().set_title('fourier analysis of on')
    plt.imshow(fou_on[fou_crop:-fou_crop, fou_crop:-fou_crop])
    plt.subplot(3,3,9)
    plt.gca().set_title('1D representation of fourier')
    plt.plot(fou_on_plot)
    plt.plot(match)
    plt.show()

    print('cosine similarity:', cos_sim)
    print('std: ', general_std)
    print('kurtosis: ', kurt_on, kurt_off)
    print('combined (std/rmse): ', general_std*rmse)
    print(X.shape)