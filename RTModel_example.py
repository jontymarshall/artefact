import numpy as np
import matplotlib.pyplot as plt
import miepython as mpy
import pathos.multiprocessing as mp
from numba import jit
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
        for ij in range(0,int(m.parameters['nring'])):
            ax.loglog(m.sed_wave,m.sed_ringe[ij,:],linestyle='-',color='orange',alpha=0.1)
    except:
        print("No continuum emission model.")
    
    try:
        ax.loglog(m.sed_wave, m.sed_scat, color='blue',linestyle=':')
        for ij in range(0,int(m.parameters['nring'])):  
            ax.loglog(m.sed_wave,m.sed_rings[ij,:],linestyle='-',color='dodgerblue',alpha=0.1    
    except:
        print("No scattered light model.")

    ax.loglog(m.sed_wave, (m.sed_emit + m.sed_scat), color='black',linestyle='--')
    ax.loglog(m.sed_wave, m.sed_star, color='black',linestyle='-.')
    
    ax.loglog(m.sed_wave, m.sed_star + m.sed_emit + m.sed_scat, color='black',linestyle='-')
    ax.set_xlabel(r'$\lambda$ ($\mu$m)')
    ax.set_ylabel(r'Flux density (mJy)')
    ax.set_xlim(m.parameters["lmin"],m.parameters["lmax"])
    if np.max(m.sed_star) > np.max((m.sed_emit + m.sed_scat)):
        ax.set_ylim(10**(np.log10(np.max((m.sed_emit + m.sed_scat))) - 6),10**(np.log10(np.max(m.sed_star)) + 1))
    else:
        ax.set_ylim(10**(np.log10(np.max(m.sed_star)) - 6),10**(np.log10(np.max((m.sed_emit + m.sed_scat))) + 1))    
    fig.savefig(m.parameters['directory']+m.parameters['prefix']+'_sed.png',dpi=200)
    plt.close(fig)

    m.figure = fig

#benchmarking with time
start = time.time()

model = RTModel()

RTModel.get_parameters(model,'RTModel_Input_File.txt')

model.parameters['mdust'] = 1e-3

RTModel.make_star(model)

RTModel.make_dust(model)

RTModel.make_disc(model)

RTModel.read_optical_constants(model)

RTModel.calculate_qabs(model)

RTModel.calculate_dust_emission(model,blackbody=False,tolerance=0.01)

RTModel.calculate_dust_scatter(model)

make_sed(model)

end = time.time()
multi_time = end - start
print("SED calculations took: ",multi_time," seconds.")