#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:42:22 2020

@author: jonty
"""

import numpy as np
import matplotlib.pyplot as plt
import miepython as mpy
import pathos.multiprocessing as mp
import time
from astropy.io import ascii
from scipy import interpolate
from scipy import integrate

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

#dust properties
#amin,amax in microns
#density in g cm^-3
dust_params = {'composition':'astrosil',
               'density':3.3,
               'amin':0.5,
               'amax':1000.0,
               'ngrain':101,
               'q':-3.5,
               'mdust':0.0001,
               'tmin':20.,
               'tmax':200.,
               'ntemp':200}

#star properties
#stype either 'spectrum' or 'blackbody'
#lstar in L_Sol
#rstar in R_Sol
#tstar in K
#dstar in pc
#Sun-like star
#star_params = {'stype':'blackbody',
#               'lstar':1.0,
#               'rstar':1.0,
#               'tstar':5770.0,
#               'dstar':10.0,
#               'model':None}
star_params = {'stype':'blackbody',
               'lstar':1.0,
               'rstar':1.0,
               'tstar':5770.0,
               'dstar':10.0,
               'model':'G2V_spectrum.txt'}
#disc properties
#dtype one of 'onepl', 'twopl', 'gauss'
#rin, rout in au; alpha is unitless
#rpeak, rwide in au
#rpeak in au; alpha_in, alpha_out are unitless
#mdust in M_Earth
disc_params = {'dtype':'onepl',
               'nring':101,
               'rpeak':50.,
               'rfwhm':10.,
               'rin':40.,
               'rout':60.,
               'alpha_in':10.,
               'alpha_out':-1}

#general model parameters
model_params = {'lmin':0.1,
                'lmax':3000.0,
                'nwav':251}

#read in stellar photosphere and scale to luminosity, or generate blackbody star
def planck_lam(wav, T):
    """
    Parameters
    ----------
    lam : float or array
        Wavelengths in metres.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    intensity : float or array
        B(lam,T) W/m^2/m/sr
    """
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

def planck_nu(freq, T):
    """
    Parameters
    ----------
    freq : float or array
        Frequencies in Hertz.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    intensity : float or array
        B(nu,T) W/m^2/Hz/sr
    """    
    a = 2.0*h*freq**3
    b = h*freq/(k*T)
    intensity = a/ ( (c**2) * (np.exp(b) - 1.0) )
    return intensity

#calculate the luminosity of the source
def calc_luminosity(rstar,tstar):
    """
    Parameters
    ----------
    rstar : float
        Radius of the star in R_sol.
    tstar : float
        Temperature of the star in Kelvin.

    Returns
    -------
    lphot : float
        L_star in L_solar
    """     
    lphot= (4.0*np.pi*sb*(rstar*rsol)**2*tstar**4) / lsol
    
    return lphot
    

def make_star(star_params,model_params):
    """
    
    Function to either create a star using a blackbody, or read in a 
    photosphere model.
    
    Parameters
    ----------
    star_params : Dictionary
         Stellar parameters.
    model_params : Dictionary
        Model parameters.

    Returns
    -------
    wavelengths : float array
        Wavelengths in microns in ascending order.
    photosphere : float array
        Photospheric flux density in mJy in ascending order.

    """
    
    
    smodel = star_params['stype'] 
        
    if smodel != 'blackbody' and smodel != 'spectrum' :
        print("Input 'stype' must be one of 'blackbody' or 'spectrum'.")

    if smodel == 'blackbody':
        lstar = star_params['lstar']
        rstar = star_params['rstar']
        tstar = star_params['tstar']
        dstar = star_params['dstar']

        lmin = model_params['lmin']
        lmax = model_params['lmax']
        nwav = model_params['nwav']
        
        wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True) #um
        photosphere = planck_lam(wavelengths*um,tstar) # W/m2/sr/m
        photosphere = np.pi * photosphere * ((rstar*rsol)/(dstar*pc))**2 # W/m2/m
        
        lphot = calc_luminosity(rstar,tstar)
        print("Stellar model has a luminosity of: ",lphot," L_sol")
        
        photosphere = (lstar/lphot)*photosphere
        
    elif smodel == 'spectrum':
        lambdas,photosphere = read_star(star_params)

        lmin = model_params['lmin']
        lmax = model_params['lmax']
        nwav = model_params['nwav']
        
        wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True)

        if np.max(wavelengths) > np.max(lambdas):
            interp_lam_arr = np.logspace(np.log10(lambdas[-1]),np.log10(1.1*wavelengths[-1]),num=nwav,base=10.0,endpoint=True)
            interp_pht_arr = photosphere[-1]*(lambdas[-1]/interp_lam_arr)**2
            photosphere = np.append(photosphere,interp_pht_arr)
            lambdas = np.append(lambdas,interp_lam_arr)
            
        func = interpolate.interp1d(lambdas,photosphere)
        photosphere = func(wavelengths)
    return wavelengths, photosphere

def read_star(star_params):
    """
    Function to read in a stellar photosphere model from the SVO database.
    
    Stellar photosphere model is assumed to have wavelengths in Angstroms,
    and flux density in erg/s/cm2/A.
    
    The function will extrapolate to the longest wavelength required in the
    model, if necessary.
    
    Parameters
    ----------
    star_params : Dictionary
         Stellar parameters.

    Returns
    -------
    wavelengths : float array
        Wavelengths in microns in ascending order.
    photosphere : float array
        Photospheric flux density in mJy in ascending order.

    
    """
    spectrum_file = star_params['model']
    
    data = ascii.read(spectrum_file,comment='#',names=['Wave','Flux'])
    
    rstar = star_params['rstar']
    dstar = star_params['dstar']
    
    wavelengths =  data['Wave'].data #Angstroms
    model_spect =  data['Flux'].data #Ergs/cm**2/s/A -> 10^-7 * 10^4 * 10^10 W/m^2/Hz/m
    
    wav_um = wavelengths * 1e-4
    flx_mjy =  model_spect * (c/wavelengths**2) * 1e-3 * ((rstar*rsol)/ (dstar*pc))**2 #coversion to Flam units
    
    return wav_um,flx_mjy

#et up power law size distribution for the dust model
def make_dust(dust_params): 
    
    amin = dust_params['amin']
    amax = dust_params['amax']
    rho  = dust_params['density']
    ngrain = dust_params['ngrain']
    mdust = dust_params['mdust']
    q = dust_params['q']
    
    grain_sizes = np.logspace(np.log10(amin),np.log10(amax),num=ngrain,base=10.0,endpoint=True)
    
    grain_numbers = (grain_sizes*um)**q
    
    grain_masses  = rho*1e3*(4./3.)*np.pi*((um*grain_sizes)**3) # kg
    disc_masses  = (grain_masses*grain_numbers)
    disc_mass_scale  = (mdust*MEarth/np.sum(disc_masses)) 
    grain_numbers = grain_numbers*disc_mass_scale 
    
    return grain_sizes, grain_numbers, grain_masses

def read_optical_constants(dust_params):
    """
    Function to read in optical constants from a text file.
    
    
    Parameters
    ----------
    dust_params : Dictionary
         Dust parameters.

    Returns
    -------
    dl : float array
        Wavelength array of dust optical constants in microns.
    dn : float array
        Real part of refractive index of dust optical constants.
    dk : float array
        Imaginary part of refractive index of dust optical constants.

    
    """    
    composition_file = dust_params["composition"]+'.lnk'
    
    data = ascii.read(composition_file,comment='#')
    
    dl = data["col1"].data
    dn = data["col2"].data
    dk = data["col3"].data
         
    return dl,dn,dk


def make_disc(disc_params):
    """
    
    Parameters
    ----------
    disc_params : Dictionary
        Disc architecture parameters.

    Returns
    -------
    scale : Float array
        Fractional contribution of the total number of dust grains in each 
        annulus/radial location.
    radii : Float array
        Radial locations for the disc emission to be calculated at.

    """
    if disc_params["dtype"] == 'gauss':
        print("Gaussian ring surface density model")
        rpeak = disc_params["rpeak"]
        rfwhm = disc_params["rfwhm"]
        nring = disc_params["nring"]
        lower = rpeak - 5.0*(rfwhm/2.355)
        upper = rpeak + 5.0*(rfwhm/2.355)
        if lower < 0.:
            lower = 0.0
        radii = np.linspace(lower,upper,num=nring,endpoint=True)
        rings = np.exp(-0.5*((radii - rpeak)/(rfwhm/2.355))**2)
        scale = rings / np.sum(rings)
        
    elif disc_params["dtype"] == 'onepl':
        print("Single power law surface density model")
        rin  = disc_params["rin"]
        rout = disc_params["rout"]
        alpha = disc_params["alpha_out"]
        nring = disc_params["nring"]
        
        radii = np.linspace(rin,rout,num=nring,endpoint=True)
        rings = (radii/rin)**alpha
        scale = rings / np.sum(rings)
        
    elif disc_params["dtype"] == 'twopl':
        print("Two power law surface density model")
        rpeak = disc_params["rpeak"]
        alpha = disc_params["alpha_in"]
        gamma = disc_params["alpha_out"]
        nring = disc_params["nring"]        
        
        rout = rpeak * (0.05**(1./gamma)) #defined as where density is 5% of peak
        
        radii = np.linspace(0.0,rout,num=nring,endpoint=True)
        rings = (radii/rpeak)**alpha
        outer = np.where(radii > rpeak)
        rings[outer] = (radii[outer]/rpeak)**gamma
        scale = rings / np.sum(rings)
    
    elif disc_params["dtype"] == 'arbit':
        print("Arbitrary density distribution has not yet been implemented.")
    
    else:
        print("Model type must be one of 'onepl','twopl', 'gauss', or 'arbit'.")
    
    
    return scale, radii


def calculate_dust_temperature(dust_params,star_params,pht,ag,qabs,radius,blackbody=False,tolerance=0.01):
    """
    Parameters
    ----------
    dust_params : Dictionary
        Disc architecture parameters.
    star_params : Dictionary
        Stellar parameters.
    pht : Float array
        Stellar photospheric emission.
    ag : Float
        Grain size in microns.
    qabs : Float array
        Absorption coefficients for grain size ag across all wavelengths.
    radius : Float
        Stellocentric distance in au.
    blackbody : True/False
        Keyword for implementing iterative dust temperature calculation.
    tolerance :
        Value to iterate towards
    
    Returns
    -------
    td : Float
        Dust temperature in Kelvin.   
    """
    
    lstar = star_params["lstar"]
    rstar = star_params["rstar"]
    tstar = star_params["tstar"]
    
    if blackbody == True:
        td = 278.*(lstar**0.25)*(radius**(-0.5))
        return td
    else:
        #temperatures = np.logspace(np.log10(dust_params["tmin"]),np.log10(dust_params["tmax"]),dust_params["ntemp"],endpoint=True)
        td = 100.0
        tstep = 50.0
        
        delta = 1e30
        nsteps = 0
        
        while delta > tolerance: 
            
            dust_absr = integrate.trapz(qabs*planck_lam(wav*um,tstar),wav*um)
            
            dust_emit = integrate.trapz(qabs*planck_lam(wav*um,td),wav*um)
            
            rdust = 2.0*((rstar*rsol)/au)*(dust_absr/dust_emit)**0.5

            delta_last = delta
            delta = abs(radius - rdust) / radius
            
            #print(delta,td,dust_absr,dust_emit,radius,rdust)                
            if radius < rdust :
                td = td + tstep
            else:
                td = td - tstep
            
            if delta > delta_last:
                tstep = tstep/2.
            
            nsteps += 1
            
            if nsteps > 50:
                print("Iterative search for best-fit temperature failed after ",nsteps," steps.")
                break
        #print(nsteps-1)
        return td
        

    
disc_params = {'dtype':'onepl',
               'nring':51,
               'rpeak':50.,
               'rfwhm':10.,
               'rin':40.,
               'rout':100.,
               'alpha_in':10.,
               'alpha_out':0.0}


#benchmarking with time
start = time.time()

wav, pht = make_star(star_params,model_params)

ag, ng, mg = make_dust(dust_params)

weights, radii = make_disc(disc_params)

#interpolate dust optical constants to same wavelength grid as star
opt_const_l,opt_const_n,opt_const_k = read_optical_constants(dust_params)
func = interpolate.interp1d(opt_const_l,opt_const_n)
dust_n = func(wav)
func = interpolate.interp1d(opt_const_l,opt_const_k)
dust_k = func(wav)

dust_nk = dust_n - 1j*np.abs(dust_k)

#calculate optical constants
# x = 2.*np.pi*ag/wav
# dust_nk = np.zeros(wav.shape,dtype='complex')
# qabs = np.zeros(wav.shape)
# for i in range(0,len(wav)):
#     dust_nk[i] = complex(dust_n[i],dust_k[i])
#     qext, qsca, qback, g = mpy.mie(dust_nk,x)
#     qabs[i] = (qext - qsca)

#array to store emission
na = dust_params["ngrain"]
nr = disc_params["nring"]
nl = model_params["nwav"]

sed_tot = np.zeros(wav.shape)
sed_ring = np.zeros((nr,nl))
sed_wav = wav

dstar = star_params["dstar"]

#loop over grain size and radius to calculate dust emission


tdust = np.zeros((na,nr))
dr = radii[1] - radii[0]
for ii in range(0,na):  
    x = 2.*np.pi*ag[ii]/wav
    qabs = np.zeros(wav.shape)
    qext, qsca, qback, g = mpy.mie(dust_nk,x)
    qabs = (qext - qsca)
    
    for ij in range(0,nr):    
        scalefactor = ng[ii]*weights[ij]*((ag[ii]*um)**2)/(dstar*pc)**2
        tdust[ii,ij] = calculate_dust_temperature(dust_params,star_params,pht,ag[ii],qabs,radii[ij],blackbody=False,tolerance=0.01)        
        sed_flx  = scalefactor * qabs * np.pi * planck_lam(sed_wav*um, tdust[ii,ij]) 
        sed_ring[ij,:] += sed_flx 
        sed_tot += sed_flx
         
        
#convert model fluxes from flam to fnu (in mJy) 
convertfactor = 1e3*1e26*(sed_wav*um)**2 /c

sed_tot = sed_tot*convertfactor
sed_ring = sed_ring*convertfactor
pht = pht*convertfactor


#plot the sed
direc = '/Users/jonty/mydata/RT_Code/'

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.loglog(sed_wav, sed_tot, color='black',linestyle='--')
ax.loglog(sed_wav, pht, color='black',linestyle='-.')
for ij in range(0,nr):
    ax.loglog(sed_wav,sed_ring[ij,:],linestyle='-',color='gray',alpha=0.1)
ax.loglog(sed_wav, sed_tot+pht, color='black',linestyle='-')
ax.set_xlabel(r'$\lambda$ ($\mu$m)')
ax.set_ylabel(r'Flux density (mJy)')
ax.set_xlim(model_params["lmin"],model_params["lmax"])
if np.max(pht) > np.max(sed_tot):
    ax.set_ylim(10**(np.log10(np.max(sed_tot)) - 4),10**(np.log10(np.max(pht)) + 1))
else:
    ax.set_ylim(10**(np.log10(np.max(pht)) - 4),10**(np.log10(np.max(sed_tot)) + 1))    
fig.savefig(direc+'RT_Code_test_sed_plot.png',dpi=200)
plt.close(fig)

end = time.time()
multi_time = end - start
print("SED calculations took: ",multi_time," seconds.")