import numpy as np
import matplotlib.pyplot as plt
import miepython.miepython as mpy
import pathos.multiprocessing as mp
import time
from astropy.io import ascii
from scipy import interpolate
from scipy import integrate

from RT_Code import RTModel

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

#plot the sed
def make_sed(m): 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    try:
        ax.loglog(m.sed_wave, m.sed_emit, color='red',linestyle=':')
        #for ij in range(0,int(m.parameters['nring'])):
        #    ax.loglog(m.sed_wave,m.sed_ringe[ij,:],linestyle='-',color='orange',alpha=0.1)
    except:
        print("No continuum emission model.")
    
    try:
        ax.loglog(m.sed_wave, m.sed_scat, color='blue',linestyle=':')
        #for ij in range(0,int(m.parameters['nring'])):  
        #    ax.loglog(m.sed_wave,m.sed_rings[ij,:],linestyle='-',color='dodgerblue',alpha=0.5)
    except:
        print("No scattered light model.")

    ax.loglog(m.sed_wave, (m.sed_emit + m.sed_scat), color='black',linestyle='--')
    ax.loglog(m.sed_wave, m.sed_star, color='black',linestyle='-.')
    ax.loglog(m.sed_wave, m.sed_star + m.sed_emit + m.sed_scat, color='black',linestyle='-')
    ax.set_xlabel(r'$\lambda$ ($\mu$m)')
    ax.set_ylabel(r'Flux density (mJy)')
    ax.set_xlim(m.parameters["lmin"],m.parameters["lmax"])
    if np.max(m.sed_star) > np.max((m.sed_emit + m.sed_scat)):
        ax.set_ylim(10**(np.log10(np.max((m.sed_emit + m.sed_scat))) - 4),10**(np.log10(np.max(m.sed_star)) + 1))
    else:
        ax.set_ylim(10**(np.log10(np.max(m.sed_star)) - 4),10**(np.log10(np.max((m.sed_emit + m.sed_scat))) + 1))    
    fig.savefig(m.parameters['directory']+m.parameters['prefix']+'_sed.png',dpi=200)
    
    plt.close(fig)

    m.figure = fig

#benchmarking with time
start = time.time()
print("Starting modelling run...")
model = RTModel()
print("Created RTModel object...")
RTModel.get_parameters(model,'RTModel_Input_File.txt')
print("Read in parameters...")
RTModel.make_star(model)
print("Created stellar model...")
RTModel.make_dust(model)
print("Created dust size distribution model...")
RTModel.make_disc(model)
print("Created disc dust distribution model...")
RTModel.calculate_surface_density(model)
print("Calculated scaling for disc annuli...")
RTModel.read_optical_constants(model)
print("Read in dust optical constants...")
RTModel.calculate_qabs(model)
print("Calculated Qabs and Qsca...")
RTModel.calculate_dust_emission(model,mode='full',tolerance=0.05)
print("Calculated dust emission...")
RTModel.calculate_dust_scatter(model)
print("Caclulated dust scattering...")
RTModel.flam_to_fnu(model)
print("Converted from Flam to Fnu...")
make_sed(model)
print("Generated SED...")
end = time.time()
multi_time = end - start
print("SED calculations took: ",multi_time," seconds.")