#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 11:42:22 2020

@author: jonty
"""

import numpy as np
import miepython as mpy
import pathos.multiprocessing as mp
from numba import jit
import time
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
        print("Instantiated radiative transfer model object.")
        self.parameters = {}
        self.sed_emit = 0.0
        self.sed_scat = 0.0
        self.sed_disc = 0.0 
        self.sed_star = 0.0
        self.sed_wave = 0.0
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
        
        
        smodel = self.parameters['stype'] 
            
        if smodel != 'blackbody' and smodel != 'spectrum' :
            print("Input 'stype' must be one of 'blackbody' or 'spectrum'.")
    
        if smodel == 'blackbody':
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
            
            lphot = RTModel.calc_luminosity(rstar,tstar)
            print("Stellar model has a luminosity of: ",lphot," L_sol")
            
            photosphere = (lstar/lphot)*photosphere
            
        elif smodel == 'spectrum':
            lambdas,photosphere = RTModel.read_star(self)
    
            lmin = self.parameters['lmin']
            lmax = self.parameters['lmax']
            nwav = int(self.parameters['nwav'])
            
            wavelengths = np.logspace(np.log10(lmin),np.log10(lmax),num=nwav,base=10.0,endpoint=True)
    
            if np.max(wavelengths) > np.max(lambdas):
                interp_lam_arr = np.logspace(np.log10(lambdas[-1]),np.log10(1.1*wavelengths[-1]),num=nwav,base=10.0,endpoint=True)
                interp_pht_arr = photosphere[-1]*(lambdas[-1]/interp_lam_arr)**2
                photosphere = np.append(photosphere,interp_pht_arr)
                lambdas = np.append(lambdas,interp_lam_arr)
                
            photosphere = np.interp(wavelengths,lambdas,photosphere)
        
        elif smodel == 'function':
            print("starfish model not yet implemented.")
        
        self.sed_wave = wavelengths 
        self.sed_star = photosphere*1e3*1e26*(self.sed_wave*um)**2 /c
        
        return wavelengths, photosphere

    
    def read_star(self):
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
        wav_um : float array
            Wavelengths in microns in ascending order.
        flx_mjy : float array
            Photospheric flux density in mJy in ascending order.
    
        
        """
        spectrum_file = self.parameters['model']
        
        data = ascii.read(spectrum_file,comment='#',names=['Wave','Flux'])
        
        rstar = self.parameters['rstar']
        dstar = self.parameters['dstar']
        
        wavelengths =  data['Wave'].data #Angstroms
        model_spect =  data['Flux'].data #Ergs/cm**2/s/A -> 10^-7 * 10^4 * 10^10 W/m^2/Hz/m
        
        wav_um = wavelengths * 1e-4
        flx_mjy =  model_spect * (c/wavelengths**2) * 1e-3 * ((rstar*rsol)/ (dstar*pc))**2 #coversion to Flam units
        
        return wav_um,flx_mjy

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
        
        grain_numbers = (grain_sizes*um)**q
        
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
            print("Gaussian ring surface density model")
            rpeak = self.parameters["rpeak"]
            rfwhm = self.parameters["rfwhm"]
            nring = int(self.parameters["nring"])
            lower = rpeak - 5.0*(rfwhm/2.355)
            upper = rpeak + 5.0*(rfwhm/2.355)
            if lower < 0.:
                lower = 0.0
            radii = np.linspace(lower,upper,num=nring,endpoint=True)
            rings = np.exp(-0.5*((radii - rpeak)/(rfwhm/2.355))**2)
            scale = rings / np.sum(rings)
            
        elif self.parameters["dtype"] == 'onepl':
            print("Single power law surface density model")
            rin  = self.parameters["rin"]
            rout = self.parameters["rout"]
            alpha = self.parameters["alpha_out"]
            nring = int(self.parameters["nring"])
            
            radii = np.linspace(rin,rout,num=nring,endpoint=True)
            rings = (radii/rin)**alpha
            scale = rings / np.sum(rings)
            
        elif self.parameters["dtype"] == 'twopl':
            print("Two power law surface density model")
            rpeak = self.parameters["rpeak"]
            alpha = self.parameters["alpha_in"]
            gamma = self.parameters["alpha_out"]
            nring = int(self.parameters["nring"])        
            
            rout = rpeak * (0.05**(1./gamma)) #defined as where density is 5% of peak
            
            radii = np.linspace(0.0,rout,num=nring,endpoint=True)
            rings = (radii/rpeak)**alpha
            outer = np.where(radii > rpeak)
            rings[outer] = (radii[outer]/rpeak)**gamma
            scale = rings / np.sum(rings)
        
        elif self.parameters["dtype"] == 'arbit':
            print("Arbitrary density distribution has not yet been implemented.")
        
        else:
            print("Model type must be one of 'onepl','twopl', 'gauss', or 'arbit'.")
        
        self.scale = scale
        self.radii = radii 
        
        #return scale, radii
    
    #@jit(nopython=True)
    def calculate_dust_temperature(self,radius,qabs,blackbody=False,tolerance=0.01):
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
                
        if blackbody == True:
            td = 278.*(lstar**0.25)*(radius**(-0.5))
            return td
        else:
            td = 100.0 #inital guess temperature- this is in the mid-range for most debris discs (~30 - 300 K)
            tstep = 50.0 #go for a big step to start with to speed things up if our inital guess is bad
            
            delta = 1e30
            #nsteps = 0
            factor = 2.0*((rstar*rsol)/au)
            dust_absr = np.trapz(qabs*RTModel.planck_lam(self.sed_wave*um,tstar),self.sed_wave*um)
            
            while delta > tolerance: 
                
                dust_emit = np.trapz(qabs*RTModel.planck_lam(self.sed_wave*um,td),self.sed_wave*um)
                
                rdust = factor*(dust_absr/dust_emit)**0.5
    
                delta_last = delta
                delta = abs(radius - rdust) / radius
                
                #print(delta,td,dust_absr,dust_emit,radius,rdust)                
                if radius < rdust :
                    td += tstep
                else:
                    td -= tstep
                
                if delta < delta_last:
                    if tstep > 0.1 : tstep = tstep/2.
                
                #nsteps += 1
                
                #if nsteps >= 50:
                #    print("Iterative search for best-fit temperature failed after ",nsteps," steps.")
                #    break
            #print(nsteps-1)
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
       
    def calculate_dust_scatter(self):
        """
        Function to calculate the scattered light contribution to the total emission from the disc.
           
        """
        
        self.sed_rings = np.zeros((int(self.parameters['nring']),int(self.parameters['nwav']))) 
        
        for ii in range(0,int(self.parameters['ngrain'])):  
            alb  = self.qsca[ii,:]/self.qext[ii,:] 
            scalefactor = self.qsca[ii,:]*alb*self.ng[ii]*((self.ag[ii]*um)**2)
            for ij in range(0,int(self.parameters['nring'])):
                scalefactor = scalefactor*self.scale[ij]/(2.*self.radii[ij]*au)**2
                self.sed_rings[ij,:] = scalefactor * self.sed_star
                self.sed_scat += scalefactor * self.sed_star
        #convert model fluxes from flam to fnu (in mJy) 
        convertfactor = 1e3*1e26*(self.sed_wave*um)**2 /c

        self.sed_rings = self.sed_rings#*convertfactor
        self.sed_scat  = self.sed_scat#*convertfactor
        self.sed_disc  += self.sed_scat   

    def calculate_dust_emission(self,blackbody=False,tolerance=0.01):
        """
        Function to calculate the continuum emission contribution to the total emission from the disc.
        
        Parameters
        ----------
        blackbody : True/False
            Keyword for implementing iterative dust temperature calculation.
        tolerance :
            Maximum allowed difference between computed radius and ring element
        
        """
        
        self.sed_ringe = np.zeros((int(self.parameters['nring']),int(self.parameters['nwav']))) 
        
        for ii in range(0,int(self.parameters['ngrain'])):  
            qabs = (self.qext[ii,:] - self.qsca[ii,:])
            for ij in range(0,int(self.parameters['nring'])):
                scalefactor = self.ng[ii]*self.scale[ij]*((self.ag[ii]*um)**2)/(self.parameters['dstar']*pc)**2
                tdust = RTModel.calculate_dust_temperature(self,self.radii[ij],qabs,blackbody=False,tolerance=0.01)
                self.sed_ringe[ij,:] = scalefactor * qabs * np.pi * RTModel.planck_lam(self.sed_wave*um, tdust)
                self.sed_emit += scalefactor * qabs * np.pi * RTModel.planck_lam(self.sed_wave*um, tdust)
        #convert model fluxes from flam to fnu (in mJy) 
        convertfactor = 1e3*1e26*(self.sed_wave*um)**2 /c

        self.sed_ringe = self.sed_ringe*convertfactor
        self.sed_emit  = self.sed_emit*convertfactor
        self.sed_disc  += self.sed_emit*convertfactor