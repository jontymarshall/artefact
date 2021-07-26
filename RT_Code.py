#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:42:22 2020

@author: jonty
"""

import numpy as np
import miepython as mpy
from numba import jit
import copy
from astropy.io import ascii

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


class RTModel:
    
    def __init__(self):
        #print("Instantiated radiative transfer model object.")
        self.parameters = {}
        self.sed_emit = 0.0
        self.sed_scat = 0.0
        self.sed_disc = 0.0 
        self.sed_star = 0.0
        self.sed_wave = 0.0
        self.obs_flux = None
        self.obs_uncs = None
        self.obs_wave = None
    
    def get_parameters(self,filename):
        """
        Parameters
        ----------
        filename : string
            Filename containing plain text name/value pairs for model values

        Returns
        -------
        parameters : Dictionary
            Dictionary of parameters for the model.

        """
        
        with open(filename) as f:
            for line in f:
                line = line.split('#', 1)[0]
                if len(line) > 0:
                    line = line.split(':')
                    #print(line)
                    try:
                        self.parameters[line[0].rstrip()] = float(line[1])
                    except:
                        line[1].strip()
                        self.parameters[line[0].rstrip()] = str(line[1][1:-1])
        return self.parameters
        
    @jit(nopython=True,cache=True)
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
    
    @jit(nopython=True,cache=True)
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
    @jit(nopython=True)
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
         
    #@jit(nopython=True,cache=True)
    def make_star(self):
        """
        Function to either create a star using a blackbody, or read in a 
        photosphere model.
        
        Returns
        -------
        wavelengths : float array
            Wavelengths in microns in ascending order.
        photosphere : float array
            Photospheric flux density in mJy in ascending order.
    
        """
        
        if self.parameters['stype'] != 'blackbody' and \
           self.parameters['stype'] != 'spectrum' and \
           self.parameters['stype'] != 'starfish':
            print("Input 'stype' must be one of 'blackbody', 'spectrum', or 'starfish'.")
    
        if self.parameters['stype'] == 'blackbody':
            lstar = self.parameters['lstar']
            rstar = self.parameters['rstar']
            tstar = self.parameters['tstar']
            dstar = self.parameters['dstar']
    
            lmin = self.parameters['lmin']
            lmax = self.parameters['lmax']
            nwav = int(self.parameters['nwav'])
            
            wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True) #um
            photosphere = RTModel.planck_lam(wavelengths*um,tstar) # W/m2/sr/m
            photosphere = np.pi * photosphere * ((rstar*rsol)/(dstar*pc))**2 # W/m2/m
            
            photosphere = photosphere*1e3*1e26
            self.sed_wave = wavelengths # um
            self.sed_star = photosphere # mJy           
                    
        elif self.parameters['stype'] == 'spectrum':
            lambdas,photosphere = RTModel.read_star(self) #returns wavelength, stellar spectrum in um, mJy
            lstar = self.parameters['lstar']
            rstar = self.parameters['rstar']
            dstar = self.parameters['dstar']
            
            lmin = self.parameters['lmin']
            lmax = self.parameters['lmax']
            nwav = int(self.parameters['nwav'])
            
            wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True)
    
            if np.max(wavelengths) > np.max(lambdas):
                interp_lam_arr = np.logspace(np.log10(lambdas[-1]),np.log10(1.1*wavelengths[-1]),num=nwav,base=10.0,endpoint=True)
                interp_pht_arr = photosphere[-1]*(lambdas[-1]/interp_lam_arr)**4
                lambdas = np.append(lambdas,interp_lam_arr)
                photosphere = np.append(photosphere,interp_pht_arr)
            
            photosphere = np.interp(wavelengths,lambdas,photosphere)
            photosphere = photosphere*1e3*1e26*((rstar*rsol)/(dstar*pc))**2
            
            self.sed_wave = wavelengths #um
            self.sed_star = photosphere #flam
            
        elif self.parameters['stype'] == 'starfish':
            print("starfish model not yet implemented.")
                
    def read_star(self):
        """
        Function to read in a stellar photosphere model from the SVO database.
        
        Stellar photosphere model is assumed to have wavelengths in Angstroms,
        and flux density in erg/s/cm^2/A.
        
        The function will extrapolate to the longest wavelength required in the
        model, if necessary.
        
        Parameters
        ----------
        star_params : Dictionary
             Stellar parameters.
    
        Returns
        -------
        wav_um : float array
            Wavelengths in microns in ascending order.
        flx_mjy : float array
            Photospheric flux density in mJy in ascending order.
    
        
        """
        spectrum_file = self.parameters['model']
        
        data = ascii.read(spectrum_file,comment='#',names=['Wave','Flux'])
        
        model_waves =  data['Wave'].data #Angstroms
        model_spect =  data['Flux'].data #Ergs/cm**2/s/A 

        model_waves = model_waves*1e-4 #um        
        model_spect = 1e-7*1e4*1e10*model_spect #W/m2/m

        
        return model_waves, model_spect

    #Scale photosphere model to observations after creation
    def scale_star(self,lmax=10.):
        
        #Observations and stellar photosphere model
        fobs = self.obs_flux[np.where(self.obs_wave <= lmax)]
        lobs = self.obs_wave[np.where(self.obs_wave <= lmax)]
        uobs = self.obs_uncs[np.where(self.obs_wave <= lmax)]
        
        smod = copy.copy(self.sed_star)*(self.sed_wave*um)**2 /c 
        lmod = self.sed_wave
        
        #interpolate model at observed wavelengths
        from scipy import interpolate
        
        f = interpolate.interp1d(lmod,smod)
        sint = f(lobs)
        
        from scipy.optimize import curve_fit
        
        def func(x,a):
            return x*a
        
        popt, pcov = curve_fit(func, sint, fobs,sigma=uobs)
        
        self.sed_star = popt[0]*self.sed_star
        
    #set up power law size distribution for the dust model
    def make_dust(self): 
        """
        Function to calculate dust grain sizes, numbers, and masses.

        Returns
        -------
        None.

        """
        amin = self.parameters['amin']
        amax = self.parameters['amax']
        rho  = self.parameters['density']
        ngrain = int(self.parameters['ngrain'])
        mdust = self.parameters['mdust']
        q = self.parameters['q']
        
        grain_sizes = np.logspace(np.log10(amin),np.log10(amax),num=ngrain,base=10.0,endpoint=True)
        
        grain_numbers = (grain_sizes)**q
        
        grain_masses  = rho*1e3*(4./3.)*np.pi*((um*grain_sizes)**3) # kg
        disc_masses  = (grain_masses*grain_numbers)
        disc_mass_scale  = (mdust*MEarth/np.sum(disc_masses)) 
        grain_numbers = grain_numbers*disc_mass_scale 
        
        self.ag = grain_sizes
        self.ng = grain_numbers
        self.mg = grain_masses
        
        #return grain_sizes, grain_numbers, grain_masses

    
    def read_optical_constants(self):
        """
        Function to read in optical constants from a text file.
        
        Returns
        -------
        dl : float array
            Wavelength array of dust optical constants in microns.
        dn : float array
            Real part of refractive index of dust optical constants.
        dk : float array
            Imaginary part of refractive index of dust optical constants.
    
        
        """    
        composition_file = self.parameters["composition"]+'.lnk'
        
        data = ascii.read(composition_file,comment='#')
        
        dl = data["col1"].data
        dn = data["col2"].data
        dk = data["col3"].data
        
        dust_n = np.interp(self.sed_wave,dl,dn)
        dust_k = np.interp(self.sed_wave,dl,dk)        
        dust_nk = dust_n - 1j*np.abs(dust_k)
        
        self.oc_nk  = dust_nk
        
        #return dl,dn,dk

    
    def make_disc(self):
        """    
        Function to make up the radial dust density distribution.
        
        Returns
        -------
        scale : Float array
            Fractional contribution of the total number of dust grains in each 
            annulus/radial location.
        radii : Float array
            Radial locations for the disc emission to be calculated at.
    
        """
        if self.parameters["dtype"] == 'gauss':
            #print("Gaussian ring surface density model")
            rpeak = self.parameters["rpeak"]
            rfwhm = self.parameters["rfwhm"]
            nring = int(self.parameters["nring"])
            lower = rpeak - 3.0*(rfwhm/2.355)
            upper = rpeak + 3.0*(rfwhm/2.355)
            if lower < 0.:
                lower = 1.0
            radii = np.linspace(lower,upper,num=nring,endpoint=True)
            rings = np.exp(-0.5*((radii - rpeak)/(rfwhm/2.355))**2)
            scale = rings / np.sum(rings)
            
        elif self.parameters["dtype"] == 'onepl':
            #print("Single power law surface density model")
            rin  = self.parameters["rin"]
            rout = self.parameters["rout"]
            alpha = self.parameters["alpha_out"]
            nring = int(self.parameters["nring"])
            
            radii = np.linspace(rin,rout,num=nring,endpoint=True)
            rings = (radii/rin)**alpha
            scale = rings / np.sum(rings)
            
        elif self.parameters["dtype"] == 'twopl':
            #print("Two power law surface density model")
            rpeak = self.parameters["rpeak"]
            alpha = self.parameters["alpha_in"]
            gamma = self.parameters["alpha_out"]
            nring = int(self.parameters["nring"])        
            
            rout = rpeak * (0.05**(1./gamma)) #defined as where density is 5% of peak
            
            radii = np.linspace(0.0,rout,num=nring,endpoint=True)
            rings = (radii/radii[0])**4 * (radii/rpeak)**alpha
            outer = np.where(radii > rpeak)
            rings[outer] = radii[outer]**2 * (radii[outer]/rpeak)**gamma
            scale = rings / np.sum(rings)
        
        elif self.parameters["dtype"] == 'arbit':
            print("Arbitrary density distribution has not yet been implemented.")
        
        else:
            print("Model type must be one of 'onepl','twopl', 'gauss', or 'arbit'.")
        
        self.scale = scale
        self.radii = radii 
        
        #return scale, radii
    
    #@jit(nopython=True)
    def calculate_dust_temperature(self,radius,qabs,mode='bb',tolerance=0.01):
        """
        Function to calculate the temperature of a dust grain at a given distance from the star.
        
        Parameters
        ----------
        radius : Float
            Element of self.radii to fit temperature for.
        qabs : Float array
            Absorption coefficients for grain size ag across all wavelengths.
        blackbody : True/False
            Keyword for implementing iterative dust temperature calculation.
        tolerance :
            Maximum allowed difference between computed radius and ring element
        
        Returns
        -------
        td : Float
            Dust temperature in Kelvin.   
        """
        
        lstar = self.parameters["lstar"]
        rstar = self.parameters["rstar"]
        tstar = self.parameters["tstar"]
        dstar = self.parameters["dstar"]
        
        if mode != 'bb' and mode != 'full':
            print("Dust temperature calculation mode must be one of 'bb' or 'full'.")
        
        if mode == 'bb':
            td = 278.*(lstar**0.25)*(radius**(-0.5))
            return td
        else:
            td = 278.*(lstar**0.25)*(radius**(-0.5)) #inital guess temperature at blackbody temperature
            tstep = 0.25*td #go for a big step to start with to speed things up if our inital guess is bad
            
            delta = 1e30
            
            factor = 0.5*((rstar*rsol)/au)
            if self.parameters["stype"] == 'blackbody':
                dust_absr= np.trapz(qabs*RTModel.planck_lam(self.sed_wave*um,tstar),self.sed_wave*um)
            elif self.parameters["stype"] == 'spectrum':
                dust_absr = np.trapz(qabs*self.sed_star*1e-26*1e-3*((dstar*pc)/(rstar*rsol))**2,self.sed_wave*um)
            
            while delta > tolerance: 
                
                dust_emit = np.trapz(qabs*RTModel.planck_lam(self.sed_wave*um,td),self.sed_wave*um)
                
                rdust = factor*(dust_absr/dust_emit)**0.5
    
                delta_last = delta
                delta = abs(radius - rdust) / radius
                
                if radius < rdust :
                    td += tstep
                else:
                    td -= tstep
                
                if delta < delta_last and tstep > 0.1:
                    tstep = tstep/2.
            
            return td
 
    def calculate_qabs(self):
        """
        Function to calculate the qabs,qsca values for the grains in the model.

        Returns
        -------
        None.

        """
        self.qext = np.zeros((int(self.parameters['ngrain']),int(self.parameters['nwav'])))
        self.qsca = np.zeros((int(self.parameters['ngrain']),int(self.parameters['nwav'])))
        
        for ii in range(0,int(self.parameters['ngrain'])):  
            x = 2.*np.pi*self.ag[ii]/self.sed_wave
            qext, qsca, qback, g = mpy.mie(self.oc_nk,x)
            
            self.qext[ii,:] = qext
            self.qsca[ii,:] = qsca
        
        self.qabs = self.qext - self.qsca
        
    def calculate_dust_scatter(self):
        """
        Function to calculate the scattered light contribution to the total emission from the disc.
           
        """
        
        self.sed_rings = np.zeros((int(self.parameters['nring']),int(self.parameters['nwav']))) 
        
        for ii in range(0,int(self.parameters['ngrain'])):  
            alb  = self.qsca[ii,:]/self.qext[ii,:] 
            scalefactor = self.qsca[ii,:]*alb*self.ng[ii]*np.pi*((self.ag[ii]*um)**2)
            
            for ij in range(0,int(self.parameters['nring'])):
                scalefactor = scalefactor*self.scale[ij]/(2.*self.radii[ij]*au)**2
                self.sed_rings[ij,:] = scalefactor * self.sed_star
                self.sed_scat += scalefactor * self.sed_star

        self.sed_rings = self.sed_rings
        self.sed_scat  = self.sed_scat
        self.sed_disc  += self.sed_scat   

    def calculate_dust_emission(self,*args,**kwargs):
        """
        Function to calculate the continuum emission contribution to the total emission from the disc.
        
        Parameters
        ----------
        mode : 'bb'/'full'
            Keyword for implementing iterative dust temperature calculation.
        tolerance :
            Maximum allowed difference between computed radius and ring element
        
        """

        #Calculate dust temperatures
        self.tdust = np.zeros((int(self.parameters["nring"]),int(self.parameters["ngrain"])))

        for ii in range(0,int(self.parameters['nring'])):
            for ij in range(0,int(self.parameters['ngrain'])):
                self.tdust[ii,ij] = RTModel.calculate_dust_temperature(self,self.radii[ii],self.qabs[ij,:],**kwargs)
        
        self.sed_ringe = np.zeros((int(self.parameters["nring"]),int(self.parameters["ngrain"]),int(self.parameters["nwav"])))
        
        self.dlam = copy.copy(self.sed_wave*um)*((np.log10(self.sed_wave[-1]) - np.log10(self.sed_wave[0]))/self.parameters["nwav"])
        self.da   = copy.copy(self.ag*um)*((np.log10(self.ag[-1]) - np.log10(self.ag[0])/self.parameters["ngrain"]))
        
        #Calculate emission
        for ii in range(1,int(self.parameters['nring'])):
            for ij in range(0,int(self.parameters['ngrain'])):

                dT = np.abs((self.tdust[ii-1,ij] - self.tdust[ii,ij]))
                
                scalefactor = (2*np.pi**2/((self.parameters['dstar']*pc)**2))*self.ng[ij]*self.scale[ii]*(1/3)*(self.ag[ij]*um)**3
 
                denom = RTModel.drdt_denom(self,ii,ij)
                numer = RTModel.drdt_numer(self,ii,ij)
                
                self.sed_ringe[ii,ij,:] = denom*numer*scalefactor*self.qabs[ij,:]*RTModel.planck_lam(self.sed_wave*um,self.tdust[ii,ij])
                self.sed_ringe[ii,ij,:] *= dT*self.dlam*self.da[ij]/(self.sed_wave*um)
                
        #self.sed_ringe = np.zeros((int(self.parameters['nring']),int(self.parameters['ngrain']),int(self.parameters['nwav'])))
        #for ii in range(0,int(self.parameters['nring'])):
        #    for ij in range(0,int(self.parameters['ngrain'])):  
        #        qabs = (self.qext[ij,:] - self.qsca[ij,:])
                #scalefactor = (2*np.pi**2/((self.parameters['dstar']*pc)**2))*qabs*self.ng[ii]*self.scale[ij]*(((1/3)*self.ag[ii]*um)**3)
        #        scalefactor = (2*np.pi**2/((self.parameters['dstar']*pc)**2))*qabs*self.ng[ij]*self.scale[ii]*(self.ag[ij]*um)**2                
        #        tdust = RTModel.calculate_dust_temperature(self,self.radii[ii],qabs,**kwargs)
        #        self.sed_ringe[ii,ij,:] = scalefactor * RTModel.planck_lam(self.sed_wave*um, tdust)
                self.sed_emit += self.sed_ringe[ii,ij,:]
        self.sed_disc += self.sed_emit
    
    def drdt_denom(self,r,s):
        
        denom = 0.5*self.radii[r]*au/np.trapz(self.qabs[s,:]*RTModel.planck_lam(self.sed_wave*um,self.tdust[r,s]),self.sed_wave*um)
        
        return denom
        
    def drdt_numer(self,r,s):
        
        numer  = ((h*c)/((self.sed_wave*um)*k*self.tdust[r,s]))
        numer *= self.qabs[s,:]*RTModel.planck_lam(self.sed_wave*um,self.tdust[r,s])
        numer /= (1. - np.e**(-1*(h*c)/(self.sed_wave*um*k*self.tdust[r,s])))
        
        numer = np.trapz(numer,self.sed_wave*um)
        
        return numer
        
    def flam_to_fnu(self):
        """
        Function to convert calculated stellar and disc emission (in F_lam units) to F_nu (in mJy)
        
        Returns
        -------
        None.

        """
        
        convert_factor = (self.sed_wave*um)**2 / c
        
        self.sed_star *= convert_factor
        
        self.sed_emit *= 1e26*convert_factor
        self.sed_ringe *= 1e26*convert_factor
        
        self.sed_scat *= convert_factor
        self.sed_rings *= convert_factor