# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:11:17 2021

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

composition = 'astrosil'



def colddisk_spec(theta):

    disk = RTModel()

    if len(theta) == 3:
        mdust, q, amin = theta
    if len(theta) == 2:
        mdust, amin = theta

    RTModel.get_parameters(disk,'RTModel_Input_File.txt')
    disk.parameters['stype'] = 'blackbody'
    disk.parameters['tstar'] = inputjson['main_results'][0]['Teff']
    disk.parameters['rstar'] = inputjson['main_results'][0]['rstar']
    dstar = inputjson['main_results'][0]['plx_arcsec']
    disk.parameters['dstar'] = (1/dstar)
    disk.parameters['lstar'] = inputjson['main_results'][0]['lstar']
    disk.parameters['composition'] = composition
    disk.parameters['dtype'] = 'gauss'

    #Dust
    disk.parameters['mdust'] = mdust
    disk.parameters['rpeak'] = inputjson['diskradius']['rin']
    disk.parameters['rfwhm'] = inputjson['diskradius']['rout']

    # disk.parameters['rin'] = inputjson['diskradius']['rout']*.75
    # disk.parameters['rout'] = inputjson['diskradius']['rout']*1.25
    if len(theta) == 3:
        disk.parameters['q'] = q
    if len(theta) == 2:
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

def warmspec_p(twarm, rwarm):
    warmdisk = RTModel()

    RTModel.get_parameters(warmdisk,'RTModel_Input_File.txt')

    warmdisk.parameters['tstar'] = twarm
    warmdisk.parameters['rstar'] = rwarm
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

def model(theta):

    mdust, q, amin = theta

    model = RTModel()

    RTModel.get_parameters(model,'RTModel_Input_File.txt')

    model.obs_flux = [x*1000 for x in y]
    model.obs_uncs = [x*1000 for x in yerr]
    model.obs_wave = x
    model.parameters['composition'] = composition

    model.parameters['prefix'] = 'test_emcee'
    model.parameters['stype'] = 'blackbody'
    model.parameters['tstar'] = inputjson['main_results'][0]['Teff']
    model.parameters['rstar'] = inputjson['main_results'][0]['rstar']
    dstar = inputjson['main_results'][0]['plx_arcsec']
    model.parameters['dstar'] = (1/dstar)
    model.parameters['lstar'] = inputjson['main_results'][0]['lstar']
    model.parameters['dtype'] = 'gauss'

    #Dust
    model.parameters['mdust'] = mdust
    model.parameters['rpeak']  = inputjson['diskradius']['rin']
    model.parameters['rfwhm']  = inputjson['diskradius']['rout']

    # model.parameters['rin'] = inputjson['diskradius']['rin'] - inputjson['diskradius']['rout']/2
    # model.parameters['rout'] = inputjson['diskradius']['rin'] + inputjson['diskradius']['rout']/2

    # model.parameters['rin'] = inputjson['diskradius']['rout']*.75
    # model.parameters['rout'] = inputjson['diskradius']['rout']*1.25

    # model.parameters['rin'] = inputjson['diskradius']['rin']
    # model.parameters['rout'] = inputjson['diskradius']['rout']

    model.parameters['q'] = q
    model.parameters['amin']  = amin
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

    model_specs.append(spec)

    return spec

def model2p(theta):

    mdust, amin = theta

    model = RTModel()

    model.obs_flux = [x*1000 for x in y]
    model.obs_uncs = [x*1000 for x in yerr]
    model.obs_wave = x


    RTModel.get_parameters(model,'RTModel_Input_File.txt')

    model.parameters['composition'] = composition
    model.parameters['directory'] = '/Users/Robin/Desktop/dust/radiative_transfer_widget-main/'
    model.parameters['prefix'] = 'test_emcee'
    model.parameters['stype'] = 'blackbody'
    model.parameters['tstar'] = inputjson['main_results'][0]['Teff']
    model.parameters['rstar'] = inputjson['main_results'][0]['rstar']
    dstar = inputjson['main_results'][0]['plx_arcsec']
    model.parameters['dstar'] = (1/dstar)
    model.parameters['lstar'] = inputjson['main_results'][0]['lstar']
    model.parameters['dtype'] = 'gauss'


    #Dust
    model.parameters['mdust'] = mdust
    model.parameters['rpeak']  = inputjson['diskradius']['rin']
    model.parameters['rfwhm']  = inputjson['diskradius']['rout']
    # model.parameters['rin'] = inputjson['diskradius']['rin'] - inputjson['diskradius']['rout']/2
    # model.parameters['rout'] = inputjson['diskradius']['rin'] + inputjson['diskradius']['rout']/2
    # model.parameters['rin'] = inputjson['diskradius']['rout']*.75
    # model.parameters['rout'] = inputjson['diskradius']['rout']*1.25

    model.parameters['q'] = -3.5
    model.parameters['amin']  = amin
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

    model_specs.append(spec)

    return spec


def lnlike(theta, x, y, yerr):

    if len(theta) == 3:
        spec = model(theta)
    if len(theta) == 2:
        spec = model2p(theta)

    if disk == 2:
        warmspec = warmspec_func_t(popt1[0])
        model_spec = spec[0], spec[1] + warmspec[1]
    if disk == 1:
        model_spec = spec

    f = interpolate.interp1d(model_spec[0], model_spec[1], fill_value="extrapolate")
    y1 = np.array([x*1000 for x in y])
    yerr1 = np.array([x*1000 for x in yerr])
    yerr_rel = y1/yerr1

    lnlike = -0.5 * np.sum((((y1 - f(x))**2)/(yerr_rel**2)))
    return lnlike

def lnprior(theta):
    if len(theta) == 3:
        mdust, q, amin = theta
    else:
        mdust, amin = theta
        q = -3.5

    if (1e-7 < mdust < 1e3 and -4 < q < -3 and 0.1 < amin < 20):
        return 0.0
    else:
        return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def run_emcee(sampler,pos,ndim,steps):
    print("\nRunning burn in")
    p0 = sampler.run_mcmc(pos, 50, progress=True)
    sampler.reset()
    print("\nRunning production")
    sampler.run_mcmc(p0, steps, rstate0=np.random.get_state(), progress=True)
    print("Done.")
    return sampler

filelist = os.listdir('processed_data/')

aaaa = filelist.index('HD 278932.json')
bbbb = filelist.index('HIP 63882.json')
cccc = filelist.index('HD 189002.json')
dddd = filelist.index('HD 31392.json')
eeee = filelist.index('HIP 74995.json')

del filelist[aaaa]
del filelist[bbbb]
del filelist[cccc]
del filelist[dddd]
del filelist[eeee]

if __name__ == '__main__':

    for x in filelist:

        filename = x
        file = open("processed_data/{:}".format(filename))
        inputjson = json.load(file)

        x = np.array(inputjson['photometry']['phot_wavelength'])
        y = np.array(inputjson['photometry']['phot_fnujy'])
        yerr = np.array(inputjson['photometry']['phot_e_fnujy'])

        manager = mp.Manager()
        model_specs = manager.list()

        if len(inputjson['main_results']) == 3:
            print('Two Component Fit')
            disk = 2
            yfit = np.array([x*1000 for x in y])
            popt, pcov = optimize.curve_fit(fit_spec, x, yfit)
            popt1, pcov1 = optimize.curve_fit(fit_spec_t, x, yfit, bounds=([100, 20], [350, 100]))
        if len(inputjson['main_results']) == 2:
            print('One Component Fit')
            disk = 1

        nwalkers = 50
        nsteps = 200
        initial = np.array([5e-3, -3.5, 5.0]) #implement log space?
        ndim = len(initial)
        p0 = [np.array(initial) for i in range(nwalkers)]

        if max(inputjson['photometry']['phot_wavelength']) > 170:
            for aa in range(nwalkers):
                p0[aa] = np.array([np.random.uniform(1e-6, 1e-2),
                                   np.random.uniform(-3.9, -3.1),
                                   np.random.uniform(1, 10)])

            data = (x, y, yerr)


            start = time.time()
            print('Running emcee sampler for {:}'.format(filename.split('.')[0]))

            with mp.Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
                results = run_emcee(sampler,p0,ndim,nsteps)

            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        else:
            initial = np.array([5e-4, 5.0])
            ndim = len(initial)
            p0 = [np.array(initial) for i in range(nwalkers)]
            for aa in range(nwalkers):
                p0[aa] = np.concatenate([np.random.normal(initial[0], 1e-4, 1),
                                        np.random.normal(initial[1], 2, 1)])
            data = (x, y, yerr)

            print("Lacking long wavelength observations")

            start = time.time()
            print('Running emcee sampler for {:}'.format(filename.split('.')[0]))

            with mp.Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data, pool=pool)
                results = run_emcee(sampler,p0,ndim,nsteps)

            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        dill.dump(sampler, open('astrosil/{:}_sampler.pickle'.format(filename.split('.')[0]),"wb"))
        with open('astrosil/{:}_flatchain.json'.format(filename.split('.')[0]), 'w') as fp:
            json.dump(sampler.flatchain.tolist(), fp, indent=4)
        dill.dump(model_specs, open('astrosil/{:}_sed.pickle'.format(filename.split('.')[0]),"wb"))


        def plotter(sampler,x=x, y=y, yerr=yerr):
            y = np.array([x*1000 for x in y])
            plt.plot(x,y, zorder=15)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Wavelength in log microns')
            plt.ylabel('Flux in log mJy')
            plt.title('{:} MCMC'.format(inputjson['id']))
            plt.errorbar(x, [x*1000 for x in y], xerr=None, yerr=yerr,marker='.',
                      linestyle='',mec='black', mfc='white', fmt='none', zorder=10)
            for specs in model_specs:
                plt.plot(specs[0], specs[1], color="r", alpha=0.05)
            plt.savefig('astrosil/{:}_plot.png'.format(filename.split('.')[0]))
            plt.close()

        def stepplot(sampler):

            if ndim == 3:
                fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
                samples = sampler.get_chain()
                labels = ["mdust", "q", "amin"]
                axes[0].set_yscale('log')

                for i in range(ndim):
                    for x in range(nwalkers):
                        if samples[:, :, 0][:, x][0] != samples[:, :, 0][:, x][-1]:
                            ax = axes[i]
                            ax.plot(samples[:, :, i][:, x], "k", alpha=0.3)
                            ax.set_xlim(0, len(samples))
                            ax.set_ylabel(labels[i])
                        else:
                            ax = axes[i]
                            ax.plot(samples[:, :, i][:, x], "r", alpha=0.3)
                            ax.set_xlim(0, len(samples))
                            ax.set_ylabel(labels[i])
            if ndim == 2:
                fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
                samples = sampler.get_chain()
                labels = ["mdust", "amin"]
                axes[0].set_yscale('log')

                for i in range(ndim):
                    for x in range(nwalkers):
                        if samples[:, :, 0][:, x][0] != samples[:, :, 0][:, x][-1]:
                            ax = axes[i]
                            ax.plot(samples[:, :, i][:, x], "k", alpha=0.3)
                            ax.set_xlim(0, len(samples))
                            ax.set_ylabel(labels[i])
                        else:
                            ax = axes[i]
                            ax.plot(samples[:, :, i][:, x], "r", alpha=0.3)
                            ax.set_xlim(0, len(samples))
                            ax.set_ylabel(labels[i])


            axes[-1].set_xlabel("Step number")
            plt.suptitle('{:} Walkers Step Plot'.format(filename.split('.')[0]))
            plt.savefig('astrosil/{:}_stepplot'.format(filename.split('.')[0]))
            plt.close()

        def cornerplot(sampler):
            flat_samples = sampler.flatchain
            if ndim == 3:
                fig = corner.corner(flat_samples, labels=["mdust", "q", "amin"], quantiles=[0.16, 0.5, 0.84])
            if ndim == 2:
                fig = corner.corner(flat_samples, labels=["mdust", "amin"], quantiles=[0.16, 0.5, 0.84])

            plt.suptitle('{:} Corner Plot'.format(filename.split('.')[0]))
            plt.savefig('astrosil/{:}_cornerplot'.format(filename.split('.')[0]))
            plt.close()

        def bestfit(sampler, x=x, y=y, yerr=yerr):
            y = [x*1000 for x in y]
            samples = sampler.flatchain
            theta_max = samples[np.argmax(sampler.flatlnprobability)]
            if ndim == 3:
                best_fit_model = model(theta_max)
            if ndim == 2:
                best_fit_model = model2p(theta_max)

            speclist = colddisk_spec(theta_max)

            if disk == 2:
                warmspecp = warmspec_p(popt1[0], popt[0])
                model_plot = best_fit_model[0], best_fit_model[1] + warmspecp[1]
            if disk == 1:
                model_plot = best_fit_model

            plt.plot(x, y, 'ro', markersize=3, label='Photometry', zorder=10, alpha=0.5)

            plt.ylim(min(best_fit_model[1])*0.75,max(best_fit_model[1])*2.5)

            plt.errorbar(x, y, xerr=None, yerr=yerr,marker='.',
                      linestyle='',mec='black', mfc='white', fmt='none', zorder=15)

            plt.plot(model_plot[0],model_plot[1],
                     label='Highest Likelihood Model', color='turquoise', zorder=5, alpha=0.7)
            if disk == 2:
                plt.plot(warmspecp[0],warmspecp[1],
                         label='Warm Disk', color='coral', linestyle='dashed')
            plt.plot(speclist[0][0],speclist[0][1],
                     label='Fitted Cold Disk', color='steelblue', linestyle='dashed')
            plt.plot(speclist[1][0],speclist[1][1],
                     label='Star', color='goldenrod', linestyle='dashed')

            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Wavelength in log microns')
            plt.ylabel('Flux in log mJy')
            plt.title('{:} Best Fit Plot'.format(filename.split('.')[0]))
            plt.savefig('astrosil/{:}_bestfitplot'.format(filename.split('.')[0]), dpi=1200)
            plt.close()


        plotter(sampler)
        stepplot(sampler)
        bestfit(sampler)
        cornerplot(sampler)



# picklefile = open('samplers/{:}_sampler.pickle'.format(filename.split('.')[0]), 'rb')
# sampler = dill.load(picklefile)



# with open('two_disks.txt') as file:
#     content = file.readlines()
#     two_disks = [x.strip() for x in content]

# temp_sdb_warm = []
# temp_sdb_cold = []

# temp_fit_warm = []
# temp_fit_cold = []

# for x in two_disks:

#     filename = x
#     file = open("processed_data/{:}".format(filename))
#     inputjson = json.load(file)

#     temp_sdb_cold.append(min(inputjson['main_results'][1]['Temp'],
#                                 inputjson['main_results'][2]['Temp']))
#     temp_sdb_warm.append(max(inputjson['main_results'][1]['Temp'],
#                                 inputjson['main_results'][2]['Temp']))

#     x = np.array(inputjson['photometry']['phot_wavelength'])
#     y = np.array(inputjson['photometry']['phot_fnujy'])
#     yerr = np.array(inputjson['photometry']['phot_e_fnujy'])

#     yp = np.array([x*1000 for x in y])
#     popt, pcov = optimize.curve_fit(fit_spec, x, yp)
#     popt1, pcov1 = optimize.curve_fit(fit_spec_t, x, yp, bounds=(20, [350, 100]))

#     temp_fit_warm.append(popt1[0])
#     temp_fit_cold.append(popt1[1])

#     bbb = fit_spec(0, popt[0], popt[1])
#     warmspec = warmspec_func(popt[0])
#     coldspec = coldspec_func(popt[1])

#     ccc = fit_spec_t(0, popt1[0], popt1[1])
#     warmspec1 = warmspec_func_t(popt1[0])
#     coldspec1 = coldspec_func_t(popt1[1])

#     starspec = starspec_func()

# samples = sampler.flatchain
# theta_max = samples[np.argmax(sampler.flatlnprobability)]


# lnlike(theta, x, y, yerr)
# lnlike(theta_max, x, y, yerr)

# theta = np.array([1e-1, -4.0, 4.0])
# spec = model(theta)
# spec_max = model(theta_max)

# model_spec = spec[0], spec[1] + warmspec1[1]
# model_spec_max = spec_max[0], spec_max[1] + warmspec1[1]

# plt.plot(x, yp, zorder=15)
# # plt.plot(spec[0], spec[1], color="b")
# plt.plot(model_spec[0], model_spec[1], color="r")
# plt.plot(model_spec_max[0], model_spec_max[1], color="b")
# plt.plot(warmspec1[0], warmspec1[1], '--')

# plt.ylim(1)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Wavelength in log microns')
# plt.ylabel('Flux in log mJy')
# plt.legend()
# plt.title('{:}'.format(filename.split('.')[0]))
# plt.savefig('test/{:}_fitplot'.format(filename.split('.')[0]), dpi=1200)
# plt.close()




# theta = [5e-2, -4.0, 15.0]
# aaa = model(theta)

# plt.plot(x, yp, zorder=15)
# plt.plot(warmspec[0], warmspec[1], zorder=15)
# plt.ylim(1)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Wavelength in log microns')
# plt.ylabel('Flux in log mJy')
# plt.title('{:}'.format(filename.split('.')[0]))



# picklefile = open('samplers/HD 125162_sampler.pickle', 'rb')
# sampler = dill.load(picklefile)

# model_specs = dill.load(picklefile, fix_imports=True, encoding="latin1")

# samples = sampler.get_chain()

# for x in samples[:, :, 0]:
#     print(x)

# bbb = aaa[:, :, 0] #[:, 3]

# temp = np.empty([])
# for x in range(len(samples)):
#     if samples[:, :, 0][:, x][0] == samples[:, :, 0][:, x][-1]:
#         samples = samples[samples != samples[:, :, 0][:, x][0]]
#         # samples = np.delete(samples, x, axis=2)
#         # np.concatenate(temp, samples[3:, :, 0][:, x])

# cc = samples[:, :, 0][:, 0]
# samples = np.delete(samples[:, :, 0], 1, axis=1)
# samples = sampler.flatchain

# theta_max  = samples[np.argmax(sampler.flatlnprobability)]
# best_fit_model = model(theta_max)

# plt.plot(x,[x*1000 for x in y])
# plt.plot(best_fit_model[0],best_fit_model[1],label='Highest Likelihood Model')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Wavelength in log microns')
# plt.ylabel('Flux in log mJy')
# plt.show()

# initial = [9e-4, -2.5, 5.0]
# x = filelist[5:6]
# file = open("processed_data/{:}".format(x[0]))
# inputjson = json.load(file)


# x = np.array(inputjson['photometry']['phot_wavelength'])
# y = np.array(inputjson['photometry']['phot_fnujy'])
# yerr = np.array(inputjson['photometry']['phot_e_fnujy'])
# model_specs = []

# ndim = len(initial)
# p0 = [np.array(initial) + 1e-4*np.random.randn(ndim) for i in range(100)]


# for a in p0:
#     print(lnprob(a, x, y, yerr))




# y = np.array([x*1000 for x in y])
# plt.plot(x,y)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Wavelength in log microns')
# plt.ylabel('Flux in log mJy')
# plt.title('{:} MCMC'.format(inputjson['id']))
# plt.errorbar(x, y, xerr=None, yerr=yerr,marker='.',
#           linestyle='',mec='black', mfc='white', fmt='none',
#           label="Photometry Uncertainty", zorder=10)
# plt.plot(spec[0], spec[1], color="r", alpha=0.1)


# start = time.time()
# with mp.ProcessPool(18) as pool:
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x,y,yerr))
#     results = run_emcee(sampler,ndim,labels,steps=nsteps,prefix=direc+"SD_Vega_TwoRing_ImageMask_",state=pos)
# dill.dump(sampler,open(direc+"vega_sampler_output_SD_Vega_TwoRing_ImageMask.p","wb"))
# #os.system('rm -rf '+direc+'compositions/silicate_d03_*')
# end = time.time()
# multi_time = end - start
# print("Multiprocessing took {0:.1f} seconds".format(multi_time))
