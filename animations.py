# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:06:05 2021

@author: Robin
"""
import numpy as np
import matplotlib.pyplot as plt
import miepython.miepython as mpy
import time
from scipy import interpolate
from scipy import optimize

from RT_Code import RTModel
import json
import os
import emcee
import corner
import dill
import multiprocessing as mp


def colddisk_spec(x):

    disk = RTModel()
    RTModel.get_parameters(disk,'RTModel_Input_File.txt')
    disk.parameters['stype'] = 'blackbody'
    disk.parameters['tstar'] = 7274
    disk.parameters['rstar'] = 1.6175997059547966
    dstar = 0.011568400000000001
    disk.parameters['dstar'] = (1/dstar)
    disk.parameters['lstar'] = 6.619591172134581
    disk.parameters['dtype'] = 'gauss'

    #Dust
    disk.parameters['mdust'] = 0.01
    disk.parameters['rpeak'] = 61.3
    disk.parameters['rfwhm'] = 195.1

    disk.parameters['q'] = -3.5
    disk.parameters['amin']  = amin
    disk.parameters['alpha_out'] = 0

    RTModel.make_star(disk)
    RTModel.make_dust(disk)
    RTModel.make_disc(disk)
    RTModel.read_optical_constants(disk)
    RTModel.calculate_qabs(disk)
    RTModel.calculate_surface_density(disk)
    RTModel.calculate_dust_emission(disk,mode='full',tolerance=0.05)
    RTModel.calculate_dust_scatter(disk)
    RTModel.flam_to_fnu(disk)
    diskspec = disk.sed_wave, disk.sed_emit + disk.sed_scat
    starspec = disk.sed_wave, disk.sed_star

    return [diskspec, starspec]


amin_range = np.arange(1, 20.5, 0.5)
q_range = np.arange(-3, -4, -0.05)

for amin in amin_range:    
    model = RTModel()
    RTModel.get_parameters(model,'RTModel_Input_File.txt')    
    model.parameters['prefix'] = 'test_emcee'
    model.parameters['stype'] = 'blackbody'
    model.parameters['tstar'] = 7274
    model.parameters['rstar'] = 1.6175997059547966
    dstar = 0.011568400000000001
    model.parameters['dstar'] = (1/dstar)
    model.parameters['lstar'] = 6.619591172134581
    model.parameters['dtype'] = 'gauss'
    
    #Dust
    model.parameters['mdust'] = 0.01
    model.parameters['rpeak']  = 61.3
    model.parameters['rfwhm']  = 195.1
    
    model.parameters['q'] = -3.5
    model.parameters['amin'] = amin
    model.parameters['alpha_out'] = 0
    
    RTModel.make_star(model)
    RTModel.make_dust(model)
    RTModel.make_disc(model)
    RTModel.read_optical_constants(model)
    RTModel.calculate_qabs(model)
    RTModel.calculate_surface_density(model)
    RTModel.calculate_dust_emission(model,mode='full',tolerance=0.05)
    RTModel.calculate_dust_scatter(model)
    RTModel.flam_to_fnu(model)
    spec = model.sed_wave, model.sed_star + model.sed_emit + model.sed_scat
    
    speclist = colddisk_spec(amin)
    
    plt.plot(spec[0], spec[1], 'r', zorder=10, alpha=0.5)
    plt.plot(speclist[0][0],speclist[0][1],
             label='Disk', color='steelblue', linestyle='dashed')
    plt.plot(speclist[1][0],speclist[1][1],
             label='Star', color='goldenrod', linestyle='dashed')
    plt.legend()
    plt.ylim(0.01,10000)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength in log microns')
    plt.ylabel('Flux in log mJy')
    plt.savefig('test/anim_{:}.jpg'.format(amin), dpi=1200)
    plt.close()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
