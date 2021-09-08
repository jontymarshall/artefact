# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:05:43 2021

@author: Robin
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import ascii

def find_nearest(array, value):     
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()  
    return idx

filelist = os.listdir('jsondata/')
file = "Marshall+2021_TableB1.csv"
csvdata = ascii.read(file)

csv_dist = csvdata.columns[1]
rout_data = csvdata.columns[14]
rout_data_uplus = csvdata.columns[15]
rout_data_uminus = csvdata.columns[16]
rin_data = csvdata.columns[17]
rin_data_uplus = csvdata.columns[18]
rin_data_uminus = csvdata.columns[19]


def processing(filename):
    file = open("jsondata/{:}".format(filename))
    input_data = json.load(file)
    
    output = dict()
    
    dist = 1/input_data['main_results'][0]['plx_arcsec']
    csvindex = find_nearest(csv_dist, dist)
    
    model_comps = input_data['model_comps']
    main_results = input_data['main_results']
    star_spec = input_data['star_spec']
    
    spitzer = input_data['spectra']
    spitzer_wav, spitzer_flux, spitzer_e = [], [], []
    for x in spitzer:
        spitzer_wav.append([a for a in x['wavelength']])
        spitzer_flux.append([a for a in x['fnujy']])
        spitzer_e.append([a for a in x['e_fnujy']])
    
    spitzer_wav = [item for sublist in spitzer_wav for item in sublist]
    spitzer_flux = [item for sublist in spitzer_flux for item in sublist]
    spitzer_e = [item for sublist in spitzer_e for item in sublist]
    
    
    if 'disk_spec' in input_data.keys():
        disk_spec = input_data['disk_spec']
    
    for x, y in enumerate(main_results):
        y['model_comps'] = model_comps[x]
    
    phot_e_fnujy_input = input_data['phot_e_fnujy'][0]
    phot_fnujy_input = input_data['phot_fnujy'][0]
    phot_wavelength_input = input_data['phot_wavelength'][0]
    phot_band_input = input_data['phot_band'][0]
    
    phot_ignore = input_data['phot_ignore'][0]
    phot_upperlim = input_data['phot_upperlim'][0]
    
    phot_e_fnujy, phot_fnujy, phot_wavelength, phot_band = [], [], [], []
    
    for x in range(len(phot_wavelength_input)):
        if phot_ignore[x] == False and phot_upperlim[x] == False:
            if phot_fnujy_input[x] > 0: #Safety check, flux should be positive
                if phot_fnujy_input[x] / phot_e_fnujy_input[x] > 3:
                    phot_e_fnujy.append(phot_e_fnujy_input[x])
                    phot_fnujy.append(phot_fnujy_input[x])
                    phot_wavelength.append(phot_wavelength_input[x])
                    phot_band.append(phot_band_input[x])
    
    phot_wavelength = [x for x in phot_wavelength if x > 3]
    phot_fnujy = phot_fnujy[-len(phot_wavelength):]
    phot_e_fnujy = phot_e_fnujy[-len(phot_wavelength):]
    phot_band = phot_band[-len(phot_wavelength):]
    
    phot_wavelength = [x for x in phot_wavelength if x < 2000]
    phot_fnujy = phot_fnujy[:len(phot_wavelength)]
    phot_e_fnujy = phot_e_fnujy[:len(phot_wavelength)]
    phot_band = phot_band[:len(phot_wavelength)]
    
    #Add spitzer data points
    points = [23, 25, 27, 30, 32, 35]

    if len(spitzer_wav) != 0:
        if max(spitzer_wav) > max(points):
            for x in points:
                spitz_idx = find_nearest(spitzer_wav, x)
                spitz_fnujy = (spitzer_flux[spitz_idx] +
                                spitzer_flux[spitz_idx-1] + 
                                spitzer_flux[spitz_idx-2] + 
                                spitzer_flux[spitz_idx+1] +
                                spitzer_flux[spitz_idx+2])/5
                spitz_fnujy_e = (spitzer_e[spitz_idx] + 
                                   spitzer_e[spitz_idx-1] + 
                                   spitzer_e[spitz_idx-2] + 
                                   spitzer_e[spitz_idx+1] +
                                   spitzer_e[spitz_idx+2])/5
                phot_wavelength.append(spitzer_wav[spitz_idx])
                phot_fnujy.append(spitz_fnujy)
                phot_e_fnujy.append(spitz_fnujy_e)

    
    #Extra observation for HD 16743
    if filename == 'HD 16743.json':
        phot_wavelength.append(1300)
        phot_fnujy.append(0.001179)
        phot_e_fnujy.append(0.0339)
    
    output['id'] = filename.split('.')[0]
    output['main_results'] = main_results
    output['star_spec'] = star_spec
    if 'disk_spec' in input_data.keys():
        output['disk_spec'] = disk_spec
        
    output['photometry'] = {'phot_band': phot_band, 'phot_wavelength':
                            phot_wavelength, 'phot_fnujy': phot_fnujy,
                            'phot_e_fnujy': phot_e_fnujy}
        
    output['diskradius'] = {'rout': rout_data[csvindex], 
                            'rout_uplus': rout_data_uplus[csvindex],
                            'rout_uminus': rout_data_uminus[csvindex],
                            'rin': rin_data[csvindex], 
                            'rin_uplus': rin_data_uplus[csvindex],
                            'rin_uminus': rin_data_uminus[csvindex]}
    
    output['spectra'] = input_data['spectra']
        
    with open('jsondata_new/{:}'.format(filename), 'w') as fp:
        json.dump(output, fp, indent=4)
    
    return output

for x, a in enumerate(filelist):
    output = processing(filelist[x])


########################Plotting

# load_file = open("processed_data/HD 102647.json",)
# output = json.load(load_file)


# plt.plot(output['photometry']['phot_wavelength'], 
#           output['photometry']['phot_fnujy'], label="Photometry")

# plt.plot(output['star_spec']['wavelength'], 
#           output['star_spec']['fnujy'], label="Star Spec")

# plt.plot(output['disk_spec']['wavelength'], 
#           output['disk_spec']['fnujy'], label="Disk Spec")

# plt.errorbar(output['photometry']['phot_wavelength'],
#               output['photometry']['phot_fnujy'],
#               xerr=None,yerr=output['photometry']['phot_e_fnujy'],marker='.',
#               linestyle='',mec='black', mfc='white', fmt='none',
#               label="Photometry Uncertainty", zorder=10)

# plt.xlim([min(output['photometry']['phot_wavelength'])*0.1,
#           max(output['photometry']['phot_wavelength'])*10])
# plt.ylim([min(output['photometry']['phot_fnujy'])*0.1,
#           max(output['photometry']['phot_fnujy'])*10])

# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('Wavelength in log microns')
# plt.ylabel('Flux in log Janskys')
# plt.title('{:} Wavelength vs. Flux Density'.format(output['id']))

# plt.savefig('{:}.png'.format(output['id']))

########################Sigma plotting

# def find_nearest(array, value):     
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()  
#     return idx

# def model_fit(output):
#     sigma = []
#     for index, wv in enumerate(output['photometry']['phot_wavelength']):
#         star_flux = output['star_spec']['fnujy'][find_nearest(output['star_spec']['wavelength'], wv)]
#         disk_flux = output['disk_spec']['fnujy'][find_nearest(output['disk_spec']['wavelength'], wv)]
        
#         if len(output['spectra']) == 2:
#             disk1_flux = output['spectra'][1]['fnujy'][find_nearest(output['spectra'][1]['wavelength'], wv)]
#             disk1_flux_diff = output['photometry']['phot_fnujy'][index] - disk1_flux
            
#         if len(output['spectra']) == 3:
#             disk1_flux = output['spectra'][1]['fnujy'][find_nearest(output['spectra'][1]['wavelength'], wv)]
#             disk1_flux_diff = output['photometry']['phot_fnujy'][index] - disk1_flux
#             disk2_flux = output['spectra'][2]['fnujy'][find_nearest(output['spectra'][2]['wavelength'], wv)]
#             disk2_flux_diff = output['photometry']['phot_fnujy'][index] - disk2_flux
        
#         star_flux_diff = output['photometry']['phot_fnujy'][index] - star_flux
#         disk_flux_diff = output['photometry']['phot_fnujy'][index] - disk_flux
        
#         if len(output['spectra']) == 1 or len(output['spectra']) == 0:
#             sigma.append(min(abs(star_flux_diff), abs(disk_flux_diff))/
#                           output['photometry']['phot_e_fnujy'][index])
#         if len(output['spectra']) == 2:
#             sigma.append(min(abs(star_flux_diff), abs(disk_flux_diff), abs(disk1_flux_diff))/
#                           output['photometry']['phot_e_fnujy'][index])
#         if len(output['spectra']) == 3:
#             sigma.append(min(abs(star_flux_diff), abs(disk_flux_diff), abs(disk1_flux_diff), abs(disk2_flux_diff))/
#                           output['photometry']['phot_e_fnujy'][index])
#     return sigma

# sigma = model_fit(output)

# # plt.ylim(-20, 20)
# plt.plot(output['photometry']['phot_wavelength'], sigma)
# plt.axhline(y=3, color='r', linestyle='--')
# plt.axhline(y=-3, color='r', linestyle='--')
# plt.xlabel('Wavelength in microns')
# plt.ylabel('Abs Sigma')
# plt.title('{:} Model Fit'.format(output['id']))


