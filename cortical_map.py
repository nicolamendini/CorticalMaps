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
        rf_std,
        rf_channels,
        rf_sparsity,
        sre_std,
        lat_sparsity,
        lat_iters,
        saturation,
        homeo_target,
        homeo_timescale,
        dtype
    ):
        
        super().__init__()
        
        self.type = dtype
        self.float_mult = 1e-1
        self.cutoff = 5
        
        # defining the lateral interaction envelopes, connections cutoffs and masks
        # number of rf units has to compensate for the integer value of the sparsity of the rfs
        adjusted_rf_std = rf_std*rf_sparsity/max(round(rf_sparsity),1) 
        rf_units = oddenise(round(adjusted_rf_std*self.cutoff))
        rf_cutoff = get_circle(rf_units, rf_units/2)
        rf_envelope = get_gaussian(rf_units,adjusted_rf_std) * rf_cutoff
        # repeat the same envelope for each channel
        rf_envelope = rf_envelope.repeat(1,rf_channels,1,1).view(1,-1,1)  
        rf_envelope /= torch.sqrt((rf_envelope**2).sum(1, keepdim=True))
        self.rf_envelope = rf_envelope.to(device).type(self.type)
        
        # short range excitation
        sre_units = oddenise(round(float(sre_std)*self.cutoff))
        sre_cutoff = get_circle(sre_units, sre_units/2)
        sre = get_gaussian(sre_units, sre_std) * sre_cutoff
        sre /= sre.sum()
        self.sre = sre.to(device).type(self.type)
        
        #long range inhibition delayer mask
        lri_std = rf_std*rf_sparsity/lat_sparsity
        # 1 is added to the cutoff to compensate for the delayed start
        lri_units = oddenise(round(lri_std*(self.cutoff+1)))
        lri_cutoff = get_circle(lri_units, lri_units/2)
        lri_mask = get_gaussian(lri_units, sre_std/lat_sparsity)
        lri_mask = (1 - (lri_mask/lri_mask.max())) 
        # long range inhibition envelope
        lri_envelope = get_gaussian(lri_units, lri_std)
        lri_envelope *= lri_mask 
        lri_envelope *= lri_cutoff
        lri_envelope = lri_envelope.view(1,-1,1)
        lri_envelope /= lri_envelope.max()
        self.lri_envelope = lri_envelope.to(device).type(self.type)        
        
        # initialising the aff weights
        init_rfs = torch.ones(sheet_units**2, rf_channels*rf_units**2,1)
        init_rfs *= rf_envelope
        init_rfs /= init_rfs.sum(1, keepdim=True) / (rf_channels*rf_units**2)
        self.rfs = nn.Parameter(init_rfs.type(self.type)) 
                
        # and the lateral inhibitory correlation weights
        init_lat = torch.ones(sheet_units**2,lri_units**2,1)
        init_lat *= lri_envelope
        init_lat /= init_lat.sum(1, keepdim=True) / lri_units**2
        self.lat_weights = nn.Parameter(init_lat.type(self.type))
                        
        # defining the variables of homeostatic adaptation
        sheet_shape = [1,1,sheet_units,sheet_units]
        thresh_init = torch.zeros(sheet_shape, device=device).type(self.type)
        self.adathresh = nn.Parameter(thresh_init,requires_grad=False)
        self.avg_acts = torch.zeros(sheet_shape, device=device).type(self.type) + homeo_target
        self.lat_mean = torch.zeros(sheet_shape, device=device).type(self.type)
        self.lat_tracker = torch.ones([lat_iters+lat_iters//2]+sheet_shape, device=device).type(self.type)
        self.last_lat = torch.zeros(sheet_shape, device=device).type(self.type)
        
        # defining the learning variables
        self.raw_aff = 0
        self.aff_cache = 0
        self.lat_correlations = 0
        self.x_tiles = 0
        
        # storing some parameters
        self.homeo_lr = 1e-3
        self.strength_lr = 1e-2
        self.lat_sparsity = lat_sparsity
        self.rf_units = rf_units
        self.sre_units = sre_units
        self.lri_units = lri_units
        self.rf_sparsity = rf_sparsity
        self.rf_channels = rf_channels
        self.sheet_units = sheet_units
        self.lat_iters = lat_iters
        self.homeo_timescale = homeo_timescale
        self.homeo_target = homeo_target
        self.saturation = saturation
        self.strength = 1
        self.lat_beta = 0.5
        self.aff_scaler = 0.5
        
        self.target_x_size = sheet_units
        # tha padding should not go below an expansion of 1
        self.target_x_size += (rf_units-1)*max(1,round(rf_sparsity))
        
        print(
            '--- model initialised with rf units: {}, sre units: {}, lri units: {}'.format(
                rf_units, 
                sre_units, 
                lri_units)
        ) 
                                
    def forward(self, x, reco_flag=False, learning_flag=True):
        
        # unpacking some values
        lat_correlations = 0
        aff = 0
        raw_aff = 0
        x_tiles = 0
        self.lat_correlations = 0
        self.lat_mean *= 0
                                 
        # ensuring that the weights stay always non negative
        with torch.no_grad():
            # float mult is used to bring the numerical values to more stable ranges
            # so instead of making avery weight average to 1, it makes it average float_mult
            self.rfs /= self.rfs.sum(1,keepdim=True) / (self.rfs.shape[1]*self.float_mult)
            self.lat_weights /= self.lat_weights.sum(1,keepdim=True) / (self.lat_weights.shape[1]*self.float_mult)
                                
        lat_weights = self.lat_weights
        rfs = self.rfs
                             
        # unpacking and detaching variables to use them in computations other than learning
        # bringing the weights back to the correct range
        if not learning_flag:
            rfs = rfs.detach()
        lat_w = lat_weights.detach() 
        #lat_w /= lat_w.shape[1]*self.float_mult     
        lat_w = lat_w * self.lri_envelope
        lat_w /= lat_w.sum(1,keepdim=True) + 1e-7
                                    
        # computing the afferent response
        if not reco_flag:
            x = F.interpolate(x, self.target_x_size, mode='bilinear')
            dilation = max(1,round(self.rf_sparsity))
            x_tiles = F.unfold(x, self.rf_units, dilation=dilation)    
            x_tiles = x_tiles.permute(2,1,0)
            raw_aff = x_tiles * rfs               
            aff = raw_aff.detach() * self.rf_envelope
            self.aff_cache = (aff * self.rf_envelope).sum(1).view(1,1,self.sheet_units,self.sheet_units)
            aff = aff.sum(1)
            aff /= (self.rf_envelope * rfs.detach()).sum(1) + 1e-7
            aff = aff.view(1,1,self.sheet_units,self.sheet_units)
            # it is convenient to bring back the rfs to their functional range here
            # because there are less multiplications to do after a summation
            #aff /= rfs.shape[1]*self.float_mult 
            raw_aff = raw_aff.sum(1).view(1,1,self.sheet_units,self.sheet_units)
            x_tiles = x_tiles * self.rf_envelope
                        
        # if feedback is coming, just take the feedback
        else:
            aff = x
                        
        lat = torch.relu(aff - self.adathresh)
        lat_pad = self.lri_units//2*self.lat_sparsity
                        
        iters = self.lat_iters//2 if reco_flag else self.lat_iters
        # reaction diffusion
        for i in range(iters):
                             
            self.lat_tracker[i + self.lat_iters*reco_flag] = lat
                
            # short range excitation is applied first
            pos_pad = self.sre.shape[-1]//2
            lat = F.pad(lat, (pos_pad,pos_pad,pos_pad,pos_pad), mode='reflect')
            lat = F.conv2d(lat, self.sre)
            
            # splitting into tiles and computing the lateral inhibitory component
            lat_tiles = F.pad(lat, (lat_pad,lat_pad,lat_pad,lat_pad), value=0)
            lat_tiles = F.unfold(lat_tiles, self.lri_units, dilation=self.lat_sparsity)
            #print(lat_tiles.shape, neg_w.shape)
            lat_tiles = lat_tiles.permute(2,0,1)
            lat_neg = torch.bmm(lat_tiles, lat_w).view(1,1,self.sheet_units,self.sheet_units)
            
            # subtracting the thresholds and applying the nonlinearities
            lat = torch.relu((lat-lat_neg)*self.strength + aff*self.aff_scaler - self.adathresh)
            lat = torch.tanh(lat) / self.saturation
                    
            self.last_lat = lat
            self.lat_mean = self.lat_mean*self.lat_beta + lat*(1-self.lat_beta)
            
            if not lat.sum():
                break
            
        if not reco_flag:
            if learning_flag:

                # learn the lateral correlations
                lat_tiles = F.pad(self.lat_mean, (lat_pad,lat_pad,lat_pad,lat_pad), value=0)
                lat_tiles = F.unfold(lat_tiles, self.lri_units, dilation=self.lat_sparsity)
                lat_tiles = lat_tiles.permute(2,0,1)
                weighted_lat_tiles = lat_tiles * self.lat_mean.view(self.sheet_units**2,1,1)
                lat_correlations = torch.bmm(weighted_lat_tiles, self.lat_weights).sum()
                self.lat_correlations = lat_correlations
                    
            # update the homeostatic variables
            self.avg_acts = self.homeo_timescale*self.avg_acts + (1-self.homeo_timescale)*self.lat_mean
            gap = (self.avg_acts-self.homeo_target)
            self.adathresh += self.homeo_lr*gap
            
            beta = lat.mean() / self.homeo_target
            self.strength -= (lat.max() - 1)*self.strength_lr*beta
                
            self.x_tiles = x_tiles
            self.raw_aff = raw_aff
            
        return lat
                   
        
    # helper functions to return the rfs and lat_weights
    def get_rfs(self):
        rfs = self.rfs.detach() * self.rf_envelope
        return rfs 
    
    
    def get_lat_weights(self):
        lat_weights = self.lat_weights.detach()
        return lat_weights / (lat_weights.shape[1]*self.float_mult)