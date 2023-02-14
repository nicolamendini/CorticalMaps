# Helper functions useful to develop Models of Cortical Maps
# Author: Nicola Mendini

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import math
import torchvision.transforms.functional as TF
from matplotlib.collections import LineCollection


# Function to compute the Laplacian Of Gaussian Operator
def get_log(size, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    t = (x + y) / (2*std**2)
    LoG = -1/(math.pi*std**2) * (1-t) * torch.exp(-t)
    LoG = LoG - LoG.mean()
    return LoG


# Function to plot a grid of images
# Takes in [N,W,W] as a shape
def plot_grid(array, rows, figsize=None, title=None):
    print('plotting shape: ', array.shape)

    if figsize:
        plt.figure(figsize=figsize)
    else:
        plt.figure(figsize=(array.shape[0]//(rows*1.5),rows//1.5))

    if title:
        plt.suptitle(title, fontsize=22)

    for i in range(rows):
        for r in range(array.shape[0]//rows):

            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            img = array[i*(array.shape[0]//rows)+r]
            plt.subplot(rows, array.shape[0]//rows, i*(array.shape[0]//rows)+r+1)
            plt.axis('off')
            plt.imshow(img)

    plt.show()
    

# Function to get a Gaussian Function
# lets you control the scale of the y axis, which is useful
# for creating edge detectors
# Also lets you control the phase between [-1,1]
def get_gaussian(size, std, yscale=1, centre_x=0, centre_y=0):
    
    distance = torch.arange(size) - size//2 - centre_x*(size//2)
    x = distance.expand(1,1,size,size)**2
    distance = torch.arange(size) - size//2 - centre_y*(size//2)
    y = (distance.expand(1,1,size,size)**2).transpose(-1,-2)*yscale
    t = (x + y) / (2*std**2)
    gaussian = 1 / (np.sqrt(2*math.pi*std**2)) * torch.exp(-t)
    gaussian /= gaussian.sum()
        
    return gaussian 


# Function to get a doughnut function which is 
# defined as a Gaussian Ring around a radius value with a certain STD
def get_doughnut(size, r, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(size,size)**2
    y = x.transpose(-1,-2)
    t = torch.sqrt(x + y)
    doughnut = torch.exp(-(t - r)**2/(2*std**2))
    doughnut /= doughnut.sum()
    return doughnut


# Function to get a circle mask with ones inside and zeros outside
def get_circle(size, r):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    circle = (x + y) < r**2
    return circle

# Function to get a Cosine Grating with a specific period and phase
def get_cos(size, period, phase):
    
    distance = torch.arange(size) - size//2
    t = distance.expand(1,1,size,size)        
    cos = torch.cos(t*torch.pi/period+phase*torch.pi)
    return cos

# Function to get a cosine wave that extends radially from the origin
def get_radial_cos(size, period):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    t = torch.sqrt(x + y)
    cos = torch.cos(t*torch.pi/period)
    return cos


# Function to get a Cauchy distribution
def get_cauchy(size, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    t = (x + y)
    cauchy = std/(std**2+t)
    cauchy /= cauchy.sum()    
    return cauchy

# Function to get a set of Orientation detectors of all angles and all phases
# Result has shape [phases, angles, W, W]
def get_orientation_detectors(size, wavelength, psteps=2, steps=50):
    
    ANGLE_END = torch.pi/2
    detectors = torch.zeros(steps, psteps, size, size)
    angles = torch.linspace(-ANGLE_END, ANGLE_END - ANGLE_END/steps, steps)
    
    # phases are anticorrelated every pi/2
    phases = torch.linspace(0, torch.pi/2, psteps)
            
    for a in range(steps):
        for p in range(psteps):

            # using gabor filters to detect the fourier-like components
            detectors[a,p] = get_gabor(size, float(angles[a]), wavelength, float(phases[p]), 1)[None,None]

    # no normalisation needed cause that's already done by the get_gabor function!

    return detectors


# Function to match a target with progressively increasing Gaussians, returns the peak and the best match
def gauss_match(target, end=None, steps=100):
    
    size = target.shape[-1]
    end = size//2 if not end else end 
    hist = torch.zeros(steps)
    vals = torch.linspace(0.1, end, steps)
    target = target / torch.sqrt((target**2).sum())

    # multiply by progressively bigger gaussians
    for i in range(steps):
        t = get_gaussian(size, vals[i])[0,0]
        t /= torch.sqrt((t**2).sum())
        hist[i] = (target * t).sum()

    peak = vals[hist.argmax()]
    return hist, vals, peak

# Function to return an Affine Grid and its Inverse to trasform the input into tiles
def get_affine_grid(input_size, rf_size, grid_size, channels, device, expansion):
    
    x1 = torch.linspace(0,input_size-rf_size*expansion,grid_size)
    y1 = torch.linspace(0,input_size-rf_size*expansion,grid_size)
    
    x2 = torch.linspace(rf_size*expansion,input_size,grid_size)
    y2 = torch.linspace(rf_size*expansion,input_size,grid_size)
    
    x1 = x1[None].expand(grid_size, -1)[None]
    y1 = y1[:,None].expand(-1, grid_size)[None]
    
    x2 = x2[None].expand(grid_size, -1)[None]
    y2 = y2[:,None].expand(-1, grid_size)[None]
    
    p = torch.cat([x2,y2,x1,y1]).view(4, -1)
    
    #from https://stackoverflow.com/questions/55373014/how-to-use-spatial-transformer-to-crop-the-image-in-pytorch
            
    theta = torch.zeros(grid_size**2,2,3).to(device)
    theta[:,0,0] = (p[0]-p[2])/input_size
    theta[:,0,2] = (p[0]+p[2])/input_size -1
    theta[:,1,1] = (p[1]-p[3])/input_size
    theta[:,1,2] = (p[1]+p[3])/input_size -1
        
    affine_grid = F.affine_grid(theta, (grid_size**2, channels, rf_size, rf_size)).to(device)
    
    theta_inv = torch.zeros(grid_size**2,3,3).to(device)
    theta_inv[:, :2] = theta
    theta_inv[:,2,2] = 1
    theta_inv = torch.inverse(theta_inv)
    affine_grid_inv = F.affine_grid(theta_inv[:,:2], (grid_size**2, channels, input_size, input_size)).to(device)
    
    return affine_grid, affine_grid_inv

# Function to generate gabor filters
# Arguments are STD, ori, wavelength, phase, elipticity
def get_gabor(size, theta, Lambda, psi, gamma):
    sigma = size/4
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = size//2
    ymax = size//2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
        * (np.cos(2 * np.pi / Lambda * x_theta + psi))
    gb = torch.tensor(gb)
    
    return gb


# Function to measure the typical distance between iso oriented map domains
# Samples a certain number of orientations given by 'precision' and returns 
# the histograms of the gaussian doughnuts that were used to fit the curve together with the peak
def get_typical_dist_fourier(orientations, grid_size, size, match_std=2, precision=10, mask=0):
    
    # R is the size of the map after removing some padding size, must be odd    
    R = grid_size//2 - size
    if grid_size%2==0:
        grid_size += 1
    
    spectrum = torch.zeros(grid_size,grid_size)
    avg_spectrum = torch.zeros(grid_size,grid_size)
    avg_peak = 0
    avg_hist = torch.zeros(grid_size//2)
    ang_range = torch.linspace(0, torch.pi-torch.pi/precision, precision)
    steps = torch.fft.fftfreq(grid_size)
    hist = torch.zeros(grid_size//2)

    # average over a number of rings, given by precision
    for i in range(precision):
        
        # compute the cosine similarity and subtract that of the opposite angle
        # this is needed to get a cleaner ring
        output = torch.cos(orientations - ang_range[i])**2
        output -= torch.cos(orientations - ang_range[i] + torch.pi/2)**2
        spectrum[size:-size,size:-size] = output[size:-size+1,size:-size+1].cpu()
             
        # compute the fft and mask it to remove the central bias
        af = torch.fft.fft2(spectrum)
        af = abs(torch.fft.fftshift(af))
        af *= ~get_circle(af.shape[-1], mask)[0,0]
        
        # use progressively bigger doughnut funtions to find the most active radius
        # which will correspond to the predominant frequency
        for r in range(grid_size//2):
    
            doughnut = get_doughnut(grid_size, r, match_std)
            prod = af * doughnut
            hist[r] = (prod).sum()
                        
        argmax_p = hist.argmax()
        peak_interpolate = steps[argmax_p]
        
        # interpolate between the peak value and its two neighbours to get a more accurate estimate
        if argmax_p+1<grid_size//2:
            base = hist[argmax_p-2]
            a,b,c = (hist[argmax_p-1]-base).abs(), (hist[argmax_p]-base).abs(), (hist[argmax_p+1]-base).abs()
            tot = a+b+c
            a /= tot
            b /= tot
            c /= tot
            peak_interpolate = a*steps[argmax_p-1] + b*steps[argmax_p] + c*steps[argmax_p+1]
            
        # add the results to the average trackers
        # 1/peak_interpolate is to convert from freq to wavelength
        avg_peak += 1/peak_interpolate
        avg_spectrum += af
        avg_hist += hist
        
    avg_peak /= precision
    spectrum /= precision
    avg_hist /= precision
    
    return avg_peak, avg_spectrum, avg_hist


# Function to count the pinwheels of a map
def count_pinwheels(orientations, grid_size, size, window=7, angsteps=100, thresh=1):
    
    pinwheels = torch.zeros(grid_size-size*2+1, grid_size-size*2+1)
    ang_range = torch.linspace(0, torch.pi/2, angsteps)
    laplacian = -get_log(5,0.25)
    
    # repeat for progressive angles at given steps
    for i in range(angsteps):
    
        # compute the cosine similarity between the units and a specific angle
        # then apply a laplacian kernel to find discontinuities
        output = torch.cos(orientations.cpu() - ang_range[i])**2
        output = F.conv2d(output[None,None], laplacian.cpu(), padding=laplacian.shape[-1]//2)[0,0]
        output = torch.relu(output)
        # average over many to reinforce discontinuities that are present for each angle
        # which hopefully will be pinwheels
        pinwheels += output[size:-size+1,size:-size+1]
        
    pinwheels /= angsteps
    
    # apply a threshold to remove the noise
    pinwheels = torch.relu(pinwheels - thresh) > 0
    
    # use a sliding window to count how many distinct discontinuities are found
    count = 0
    pinwheels_copy = F.pad(pinwheels.clone(),(window//2,window//2,window//2,window//2))
    mask = torch.ones(window, window)
    mask[1:-1, 1:-1] -= torch.ones(window-2, window-2)
    skip = False

    for x in range(grid_size-size*2):
        for y in range(grid_size-size*2):
        
            curr_slice = pinwheels_copy[x:x+window, y:y+window]
        
            # if there is anything in the current window
            if curr_slice.sum():
        
                # and there is no activation on the borders of the window
                # eg: a pinwheel was fully within the window
                if (not skip) and (not (curr_slice*mask).sum()):
                
                    # then increase the counter and remove it from the discontinuity map
                    count +=1
                    pinwheels_copy[x:x+window, y:y+window] *= False
                    skip = True
        
            else:
                skip = False
                
    return count, pinwheels, pinwheels_copy

# funtion to implement batch cosine similarity
def cosine_sim(X_target, X_eff):
    
    cos_sim = (X_target*X_eff).sum()
    cos_sim /= torch.sqrt((X_target**2).sum()) * torch.sqrt((X_eff**2).sum())
    return cos_sim

# Normalised Mean Absolute Error
def NMAE(X_target, X_eff, mask=None):
    
    X_target = X_target / X_target.sum([1,2,3],keepdim=True)
    X_eff = X_eff / X_eff.sum([1,2,3],keepdim=True)
    n = X_target.shape[0]
    if mask.any():
        X_target = X_target[mask]
        X_eff = X_eff[mask]
    nmae = (X_target - X_eff).abs().sum() / n
    return nmae

# Function to return the Gini coefficient
def get_gini(X_eff):
    
    X_eff = X_eff.view(X_eff.shape[0], -1).type(torch.double)
    sorted_x = torch.sort(X_eff)[0]
    n = X_eff.shape[-1]
    cumx = torch.cumsum(sorted_x, dim=1)
    gini = torch.sum(cumx, 1) / cumx[:,-1]
    gini = (n + 1 - 2 * gini) / n
    gini = gini.mean()
    return gini

# function to get the orientation map for plotting out of a set of receptive fields
def get_orientations(rfs, ksize, grid_size, on_off_flag=True, det_steps=2, discreteness=15):
    
    # get the gabor detectors
    detectors = get_orientation_detectors(ksize,ksize,det_steps,discreteness).to(rfs.device)
    #rfs = rfs / torch.sqrt((rfs**2).sum(1, keepdim=True))
    rfs = rfs / rfs.sum(1, keepdim=True)
    rfs = rfs.view(-1, 2, ksize, ksize)

    # decide whether to measure the on or off channel
    affinity = 0
    if on_off_flag:
        affinity = (rfs[:,0,None,None]*detectors[None]).sum([-1,-2])
    else:
        affinity = (rfs[:,1,None,None]*detectors[None]).sum([-1,-2])

    # the phases are the arctan of the ratio between sin and cos
    phases = torch.arctan(affinity[:,:,1]/(affinity[:,:,0]+1e-11))
    magnitudes = torch.sqrt((affinity**2).sum(2))

    # the orientations are the gabor filters which responses had the highest magnitude
    orientations = magnitudes.max(1)[1]
    phases = phases.gather(1, orientations[:,None]).view(grid_size,grid_size)
    orientations = orientations.view(grid_size,grid_size) / orientations.max() * torch.pi
    return orientations, phases
    
# function to return some random indices to create random sparse connections
# it works but unfortunately it is very slow
def get_jittered_indices(input_size, kernel_size, dilation, output_size, channels):
    
    # a sheet of [1,1,M,N] is indexed by using two set of indices of shape [B, KSIZE^2, 1] being x and y
    rng = torch.arange(kernel_size)*dilation
    rng_row = rng.repeat_interleave(kernel_size)
    rng_row = rng_row[None,None].repeat(output_size,output_size,channels,1)
    rng_row += torch.randint(dilation,rng_row.shape) - dilation//2
    rng_row += torch.arange(output_size).view(-1,1,1,1)
    rng_row[rng_row<0] = 0
    rng_row[rng_row>=input_size] = input_size-1
    
    rng_col = rng.repeat(kernel_size)
    rng_col = rng_col[None,None].repeat(output_size,output_size,channels,1)
    rng_col += torch.randint(dilation,rng_col.shape) - dilation//2
    rng_col += torch.arange(output_size).view(1,-1,1,1)
    rng_col[rng_col<0] = 0
    rng_col[rng_col>=input_size] = input_size-1
    
    return rng_row, rng_col

# magnitude loss to ensure weights add up to 0
def get_norm_loss(weights):
    
    w_det = weights.detach()
    w_sum = w_det.sum(1, keepdim=True)
    # similar to l2 loss in that the decay is proportional to the weight but has target norn 1
    w_norm = (((w_sum - 1) * w_det * w_det.shape[1]) * weights).sum()
    return w_norm

# ensemble loss
def cosine_loss(raw_aff, lat, lat_correlations, model):
    
    #winners = torch.einsum('abcd, abcd ->',h,lat)
    winners = (raw_aff * model.lat_mean).sum()
    rfs_norm = get_norm_loss(model.rfs)    
    lat_w_norm = get_norm_loss(model.lat_weights)
    loss = -winners -lat_correlations +rfs_norm +lat_w_norm
        
    return loss

def plot_absolute_phases(model,target_channel=0):

    # exctracting useful params
    rfs = model.get_rfs().cpu()
    aff_units = model.aff_cf_units
    sheet_units = model.sheet_units
    channels = model.aff_cf_channels
    
    # making a meshgrid to localise any points within the aff cf
    rng = torch.arange(aff_units) - aff_units//2
    coordinates = torch.meshgrid(rng,rng)
    coordinates = torch.stack(coordinates)[None]
    
    # averaging over all locations to detect the greatest intensity
    rfs = rfs.view(-1,channels,aff_units,aff_units)[:,target_channel][:,None]
    rfs = rfs.repeat(1,2,1,1)
    c = (coordinates * rfs)
    c = c.sum([2,3]) * 2

    # organising everything into a grid and plotting the centre of mass of each point
    rng = torch.arange(sheet_units)
    topography = torch.meshgrid(rng,rng)
    topography = torch.stack(topography)
    topography = topography.reshape(2,-1)
    topography = (topography.T.float() + c).T

    plt.figure(figsize=(30,30))
    plt.scatter(topography[0],topography[1])

    # plotting the lines of the grid
    topography = topography.T.view(sheet_units,sheet_units,2)
    segs1 = topography
    segs2 = segs1.permute(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
    
    plt.show()

