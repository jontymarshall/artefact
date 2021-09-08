# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:29:34 2021

@author: Robin
"""
import numpy as np
import matplotlib.pyplot as plt
import miepython.miepython as mpy
import pathos.multiprocessing as pmp
import time
from scipy import interpolate
from scipy import optimize

from RT_Code import RTModel
import json
import os
import emcee
import corner
import dill
import multiprocess as mp

os.environ["OMP.NUM.THREADS"] = "1"

#constants
h = 6.626e-34
c = 299792458.0 # m/s
k = 1.38e-23    
sb = 5.67e-8 # 
au     = 1.495978707e11 # m 
pc     = 3.0857e16 # m
lsol   = 3.828e26 # W
rsol   = 6.96342e8 # m
MEarth = 5.97237e24 # kg

um = 1e-6 #for wavelengths in microns


def warmspec_func(rwarm):
    warmdisk = RTModel()
    
    RTModel.get_parameters(warmdisk,'RTModel_Input_File.txt')
    
    warmdisk.parameters['tstar'] = max(inputjson['main_results'][1]['Temp'],
                                        inputjson['main_results'][2]['Temp'])
        
    warmdisk.parameters['rstar'] = rwarm
    RTModel.make_star(warmdisk)
    warmdisk.sed_emit = 0
    warmdisk.sed_scat = 0
    warmdisk.sed_ringe = 0
    warmdisk.sed_rings = 0
    RTModel.flam_to_fnu(warmdisk)
    warmspec = warmdisk.sed_wave, warmdisk.sed_star
    
    return warmspec

def coldspec_func(rcold):
    colddisk = RTModel()
    
    RTModel.get_parameters(colddisk,'RTModel_Input_File.txt')
    
    colddisk.parameters['tstar'] = min(inputjson['main_results'][1]['Temp'],
                                        inputjson['main_results'][2]['Temp'])
    
    colddisk.parameters['rstar'] = rcold
    RTModel.make_star(colddisk)
    colddisk.sed_emit = 0
    colddisk.sed_scat = 0
    colddisk.sed_ringe = 0
    colddisk.sed_rings = 0
    RTModel.flam_to_fnu(colddisk)
    coldspec = colddisk.sed_wave, colddisk.sed_star 
    
    return coldspec

def warmspec_func_t(twarm):
    warmdisk = RTModel()
    
    RTModel.get_parameters(warmdisk,'RTModel_Input_File.txt')
    
    warmdisk.parameters['tstar'] = twarm
    warmdisk.parameters['rstar'] = popt[0]
    RTModel.make_star(warmdisk)
    warmdisk.sed_emit = 0
    warmdisk.sed_scat = 0
    warmdisk.sed_ringe = 0
    warmdisk.sed_rings = 0
    RTModel.flam_to_fnu(warmdisk)
    warmspec = warmdisk.sed_wave, warmdisk.sed_star
    
    return warmspec

def coldspec_func_t(tcold):
    colddisk = RTModel()
    
    RTModel.get_parameters(colddisk,'RTModel_Input_File.txt')
    
    colddisk.parameters['tstar'] = tcold
    colddisk.parameters['rstar'] = popt[1]
    RTModel.make_star(colddisk)
    colddisk.sed_emit = 0
    colddisk.sed_scat = 0
    colddisk.sed_ringe = 0
    colddisk.sed_rings = 0
    RTModel.flam_to_fnu(colddisk)
    coldspec = colddisk.sed_wave, colddisk.sed_star 
    
    return coldspec


def starspec_func():
    starspec = RTModel()
    RTModel.get_parameters(starspec,'RTModel_Input_File.txt')
    starspec.parameters['tstar'] = inputjson['main_results'][0]['Teff']
    starspec.parameters['rstar'] = inputjson['main_results'][0]['rstar']
    dstar = inputjson['main_results'][0]['plx_arcsec']
    starspec.parameters['dstar'] = (1/dstar)
    starspec.parameters['lstar'] = inputjson['main_results'][0]['lstar']
    RTModel.make_star(starspec)
    starspec.sed_emit = 0
    starspec.sed_scat = 0
    starspec.sed_ringe = 0
    starspec.sed_rings = 0
    RTModel.flam_to_fnu(starspec)
    sspec = starspec.sed_wave, starspec.sed_star 
    
    return sspec


def fit_spec(a, rwarm, rcold):
        
    sspec = starspec_func()
    coldspec = coldspec_func(rcold) 
    warmspec = warmspec_func(rwarm)    
       
    
    spec = warmspec[0], warmspec[1] + coldspec[1] + sspec[1]
    f = interpolate.interp1d(spec[0], spec[1], fill_value="extrapolate")
    
    return f(x)

def fit_spec_t(a, twarm, tcold):
    
    sspec = starspec_func()
    warmspec = warmspec_func_t(twarm) 
    coldspec = coldspec_func_t(tcold) 
    
    spec = warmspec[0], warmspec[1] + coldspec[1] + sspec[1]
    f = interpolate.interp1d(spec[0], spec[1], fill_value="extrapolate")
    
    return f(x)



with open('two_disks.txt') as file:
    content = file.readlines()
    two_disks = [x.strip() for x in content]

temp_sdb_warm = []
temp_sdb_cold = []

temp_fit_warm = []
temp_fit_cold = []

lnlike_sdb = []
lnlike_fit = []


for x in two_disks:
    
    filename = x
    file = open("processed_data/{:}".format(filename))
    inputjson = json.load(file)
    
    temp_sdb_cold.append(min(inputjson['main_results'][1]['Temp'],
                                inputjson['main_results'][2]['Temp']))
    temp_sdb_warm.append(max(inputjson['main_results'][1]['Temp'],
                                inputjson['main_results'][2]['Temp']))

    x = np.array(inputjson['photometry']['phot_wavelength'])
    y = np.array(inputjson['photometry']['phot_fnujy'])
    yerr = np.array(inputjson['photometry']['phot_e_fnujy'])

    yp = np.array([x*1000 for x in y])
    yerr1 = np.array([x*1000 for x in yerr])   
    
    popt, pcov = optimize.curve_fit(fit_spec, x, yp)
    popt1, pcov1 = optimize.curve_fit(fit_spec_t, x, yp, bounds=([100, 20], [350, 100]))

    temp_fit_warm.append(popt1[0])
    temp_fit_cold.append(popt1[1])
    
    sdb_warm = warmspec_func_t(max(inputjson['main_results'][1]['Temp'],
                                inputjson['main_results'][2]['Temp']))
    sdb_cold = coldspec_func_t(min(inputjson['main_results'][1]['Temp'],
                            inputjson['main_results'][2]['Temp']))
    fit_warm = warmspec_func_t(popt1[0])
    fit_cold = coldspec_func_t(popt1[1])
    starspec = starspec_func()
    
    sdb_sed = starspec[0], starspec[1] + sdb_warm[1] + sdb_cold[1]
    fit_sed = starspec[0], starspec[1] + fit_warm[1] + fit_cold[1]
    
    fsdb = interpolate.interp1d(sdb_sed[0], sdb_sed[1], fill_value="extrapolate")
    ffit = interpolate.interp1d(fit_sed[0], fit_sed[1], fill_value="extrapolate")
    
    
    sdb_lnlike = -0.5 * np.sum((((yp - fsdb(x))**2)))
    fit_lnlike = -0.5 * np.sum((((yp - ffit(x))**2)))
    lnlike_sdb.append(sdb_lnlike)
    lnlike_fit.append(fit_lnlike)
    
    
    
    plt.plot(x, yp, label='Photometry', zorder = 100)
    plt.plot(sdb_sed[0], sdb_sed[1], "r--", alpha = 0.5, label='SDB SED')
    plt.plot(fit_sed[0], fit_sed[1], "b--", alpha = 0.5, label='Fit SED')
    plt.plot(sdb_warm[0], sdb_warm[1], alpha = 0.5, label='SDB Warm Disk', color = 'brown', linestyle='dotted')
    plt.plot(sdb_cold[0], sdb_cold[1], alpha = 0.5, label='SDB Cold Disk', color = 'teal', linestyle='dotted')
    plt.plot(fit_warm[0], fit_warm[1], alpha = 0.5, label='Fit Warm Disk', color = 'darkred', linestyle='dashed')
    plt.plot(fit_cold[0], fit_cold[1], alpha = 0.5, label='Fit Cold Disk', color = 'green', linestyle='dashed')

    plt.ylim(1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength in log microns')
    plt.ylabel('Flux in log mJy')
    plt.legend()
    plt.title('{:} Disk SDB vs Fit'.format(filename.split('.')[0]))
    plt.savefig('diskfits/{:}_fitplot'.format(filename.split('.')[0]), dpi=1200)
    plt.close()

    

plt.scatter(range(len(two_disks)), temp_sdb_cold, label='SDB Cold Disk', alpha = 0.5, color = 'cyan')
plt.scatter(range(len(two_disks)), temp_sdb_warm, label='SDB Warm Disk', alpha = 0.5, color = 'olive')
plt.scatter(range(len(two_disks)), temp_fit_cold, label='Fit Cold Disk', alpha = 0.5, color = 'green')
plt.scatter(range(len(two_disks)), temp_fit_warm, label='Fit Warm Disk', alpha = 0.5, color = 'coral')
plt.legend()
plt.xlabel('File')
plt.ylabel('Temperature in K')
plt.title('Disk Parameters Fit Comparison')
plt.savefig('diskfits/diskfitdistribution', dpi=1200)




delta_lnlike = [x1 - x2 for (x1, x2) in zip(lnlike_fit, lnlike_sdb)]

plt.scatter(range(len(two_disks)), delta_lnlike, color = 'black')
plt.xlabel('File')
plt.ylabel('Fit Improvement')
plt.yscale('log')
plt.title('Disk Parameters Fit Comparison Improvement Over SDB')
plt.savefig('diskfits/diskfitcomparison', dpi=1200)












