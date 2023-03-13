# Author: Nicola Mendini
# Main cortical map module

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
import math
import torchvision.transforms.functional as TF

from maps_helper import *

# Cortical Map Module
class CorticalMap(nn.Module):
    
    def __init__(
        self, 
        device,
        sheet_units,
        aff_cf_units,
        aff_cf_channels,
        aff_cf_dilation,
        lat_cf_dilation,
        lat_iters,
        lat_strength_target,
        homeo_target,
        homeo_timescale,
        exc_units,
        lat_cf_units,
        inh_exc_balance
    ):
        
        super().__init__()
        
        # defining connections cutoffs and masks
        inh_cutoff = get_circle(lat_cf_units, lat_cf_units/2)    
        exc_cutoff = get_circle(exc_units, exc_units/2)
        aff_cutoff = get_circle(aff_cf_units, aff_cf_units/2)
        inh_mask = get_radial_cos(lat_cf_units, exc_units/lat_cf_dilation)**2
        inh_mask *= get_circle(lat_cf_units, exc_units/(2*lat_cf_dilation))
        
        self.max_rad = oddenise(exc_units*2+1)
        max_rad_mask = get_circle(self.max_rad, self.max_rad/2).view(1,-1,1)
        self.max_rad_mask = max_rad_mask.to(device)
        max_gauss = get_radial_cos(self.max_rad, self.max_rad) 
        max_gauss = max_gauss.view(1,-1,1) * max_rad_mask
        self.max_gauss = max_gauss.to(device)
        #plt.imshow(max_rad_mask.view(self.max_rad,self.max_rad))
        #plt.show()
        
        # defining the lateral interaction envelopes
        aff_cf_envelope = get_radial_cos(aff_cf_units,aff_cf_units)**2 * aff_cutoff
        aff_cf_envelope /= aff_cf_envelope.max()
        aff_cf_envelope = aff_cf_envelope.repeat(1,aff_cf_channels,1,1).view(1,-1,1)  
        self.aff_cf_envelope = aff_cf_envelope.to(device)
        
        sre = get_radial_cos(exc_units, exc_units) * exc_cutoff
        self.sre = sre.to(device)
        
        lri_envelope = get_radial_cos(lat_cf_units, lat_cf_units)
        lri_envelope *= 1 - inh_mask       
        lri_envelope *= inh_cutoff
        lri_envelope = lri_envelope.view(1,-1,1)
        lri_envelope /= lri_envelope.max()
        self.lri_envelope = lri_envelope.to(device)
        
        # initialising the aff weights
        init_rfs = torch.rand(sheet_units**2, aff_cf_channels*aff_cf_units**2, 1)
        init_rfs /= init_rfs.sum(1, keepdim=True)
        self.rfs = nn.Parameter(init_rfs) 
                
        # and the lateral inhibitory correlation weights
        init_lat = torch.ones(1,1,lat_cf_units,lat_cf_units)
        init_lat = init_lat.view(1,lat_cf_units**2,1).repeat(sheet_units**2,1,1)
        init_lat /= init_lat.sum(1, keepdim=True)
        self.lat_weights = nn.Parameter(init_lat)
        
        # defining the variables of homeostatic adaptation
        thresh_init = torch.zeros((1,1,sheet_units,sheet_units))
        self.adathresh = nn.Parameter(thresh_init,requires_grad=False)
        self.avg_acts = torch.zeros((1,1,sheet_units,sheet_units), device=device) + homeo_target
        self.lat_mean = torch.zeros((1,1,sheet_units,sheet_units), device=device)
        self.strength = 10
        
        # storing some parameters
        self.homeo_lr = (1-homeo_timescale)/2
        self.lat_cf_units = lat_cf_units
        self.lat_cf_dilation = lat_cf_dilation
        self.aff_cf_units = aff_cf_units
        self.aff_cf_dilation = aff_cf_dilation
        self.aff_cf_channels = aff_cf_channels
        self.sheet_units = sheet_units
        self.lat_iters = lat_iters
        self.homeo_timescale = homeo_timescale
        self.homeo_target = homeo_target
        self.lat_strength_target = 0.9
        self.strength_lr = 1e-2
        self.inh_exc_balance = inh_exc_balance
                                
    def forward(self, x, reco_flag=False, learning_flag=True, print_flag=False):
        
        # unpacking some values
        lat_correlations = 0
        raw_aff = 0
        x_tiles = 0
                
        # ensuring that the weights stay always non negative
        with torch.no_grad():
            self.rfs *= (self.rfs>0)
            self.lat_weights *= (self.lat_weights>0)
        
        lat_w = self.lat_weights.detach()
        rfs = self.rfs
        if not learning_flag:
            rfs = rfs.detach()
                                    
        # computing the afferent response
        if not reco_flag:
            self.lat_mean *= 0
            dilation = 1 if self.aff_cf_dilation<1 else round(self.aff_cf_dilation)
            x_tiles = F.unfold(x, self.aff_cf_units, dilation=dilation)    
            x_tiles = x_tiles * self.aff_cf_envelope
            x_tiles = x_tiles.permute(2,0,1)
            raw_aff = torch.bmm(x_tiles, rfs)
            raw_aff = raw_aff.view(1,1,self.sheet_units,self.sheet_units)
            aff = raw_aff.detach()
            
        # if feedback is coming, just take the feedback
        else:
            aff = x
            
        aff = torch.relu(aff - self.adathresh)
        lat = aff
        
        if print_flag:
            plt.imshow(lat.cpu()[0,0])
            plt.show()
               
        for i in range(15):

            pad = self.max_rad//2
            lat_tiles = F.unfold(lat, self.max_rad, padding=pad)
            lat_tiles = lat_tiles * self.max_rad_mask.float()
            voting_credits = lat.view(1,1,-1) > 0
            lat_max = lat_tiles.max(1)
            lat_max_oh = F.one_hot(lat_max[1], num_classes=self.max_rad**2)
            lat_max = lat_max_oh.permute(0,2,1).float() * voting_credits
            
            lat_max = F.fold(lat_max, output_size=self.sheet_units, kernel_size=self.max_rad, padding=pad)
            lat_max[lat_max>1] = 1
            lat = torch.relu(lat_max * aff - self.adathresh)
            lat = torch.tanh(lat * self.strength)

        lat = F.conv_transpose2d(lat, self.sre, padding=self.sre.shape[-1]//2)
        self.lat_mean = lat
        
        if print_flag:
            plt.imshow(lat.cpu()[0,0])
            plt.show()


        if not reco_flag:
            if learning_flag:

                # learn the lateral correlations
                pad = self.lat_cf_units//2*self.lat_cf_dilation
                lat_tiles = F.pad(lat, (pad,pad,pad,pad), value=self.homeo_target)
                lat_tiles = F.unfold(lat_tiles, self.lat_cf_units, dilation=self.lat_cf_dilation)
                lat_tiles = lat_tiles * self.lri_envelope 
                lat_tiles = lat_tiles.permute(2,0,1)
                weighted_lat_tiles = lat_tiles * lat.view(self.sheet_units**2,1,1)
                lat_correlations = torch.bmm(weighted_lat_tiles, self.lat_weights).sum()
                                
            # update the homeostatic variables
            self.avg_acts = self.homeo_timescale*self.avg_acts + (1-self.homeo_timescale)*self.lat_mean
            gap = (self.avg_acts-self.homeo_target)
            self.adathresh += self.homeo_lr*gap
            
            curr_max_lat = lat.max()
            #if curr_max_lat > 0.1:
            self.strength -= (curr_max_lat - self.lat_strength_target)*self.strength_lr
                                                                                
        return raw_aff, lat, lat_correlations, x_tiles
    
    def get_rfs(self):
        return self.rfs.detach()
    
    def get_lat_weights(self):
        return self.lat_weights.detach()