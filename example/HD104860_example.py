import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from artefact import RTModel

#plot the sed
def make_sed(m): 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
            
    m.sed_emit = np.nan_to_num(m.sed_emit)
    m.sed_ringe = np.nan_to_num(m.sed_ringe)
    
    try:
        ax.loglog(m.sed_wave, m.sed_emit, color='red',linestyle=':')
        #for ii in range(0,int(m.parameters['nring'])):
        #    for ij in range(0,int(m.parameters['ngrain'])):
        #        ax.loglog(m.sed_wave,m.sed_ringe[ii,ij,:],linestyle='-',color='orange',alpha=0.1)
    except:
        print("No continuum emission model.")
    
    try:
        ax.loglog(m.sed_wave, m.sed_scat, color='blue',linestyle=':')
        for ij in range(0,int(m.parameters['nring'])):  
            ax.loglog(m.sed_wave,m.sed_rings[ij,:],linestyle='-',color='dodgerblue',alpha=0.5)
    except:
        print("No scattered light model.")

    ax.loglog(m.sed_wave, (m.sed_emit + m.sed_scat), color='black',linestyle='--')
    ax.loglog(m.sed_wave, m.sed_star, color='black',linestyle='-.')
    try:
        ax.loglog(m.sed_wave, m.sed_star2, color='red',linestyle='-.')
    except:
        pass
    ax.loglog(m.sed_wave, m.sed_star + m.sed_emit + m.sed_scat, color='black',linestyle='-')
    ax.errorbar(m.obs_wave,m.obs_flux,xerr=None,yerr=m.obs_uncs,linestyle='',marker='o',mec='black',mfc='white')
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

#Target details
target = "HD104860"

#Photometry obtained using PySED to search selected Vizier catalogues
wav, flx, unc  = np.loadtxt(target+'_photometry.txt',delimiter=',',
                            comments="#", usecols=(0, 1, 2), unpack=True)

#benchmarking with time
start = time.time()

model = RTModel()

RTModel.get_parameters(model,target+'_parameters.txt')

model.obs_flux = flx
model.obs_uncs = unc
model.obs_wave = wav

RTModel.make_star(model)
RTModel.scale_star(model,lmin=0.45,lmax=5.0)
RTModel.make_dust(model)
RTModel.make_disc(model)
RTModel.calculate_surface_density(model)
RTModel.read_optical_constants(model)
RTModel.calculate_qabs(model)
RTModel.calculate_dust_emission(model,mode="full",tolerance=1e-3)
RTModel.calculate_dust_scatter(model)
RTModel.flam_to_fnu(model)
print("Ran artefact model for target: ",target)

make_sed(model)
print("Generated SED...")
end = time.time()
duration = end - start
print("SED calculations took: ",duration," seconds.")

end = time.time()
multi_time = end - start
print("SED calculations took: ",multi_time," seconds.")
