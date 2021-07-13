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
    
    wav = np.asarray([1.24,1.65,2.16,3.35,3.6,4.5,4.6,8.,11.6,13.,22.1,24.,33.,70.,100.,160.,350.,450.,850.,1200.,1300.])
    flx = np.asarray([2940.,2450.,1670.,806.,725.,455.,442.,163.,74.,57.,23.,20.,18.,183.,277.,243.,50.,47.,6.8,4.4,5.5])
    unc = np.asarray([50,40,30,32,16,11,9,4,1,4,1,1,2,15,4,5,14,18,1,1,2])
    
    print(wav.size,flx.size,unc.size)
    
    try:
        ax.loglog(m.sed_wave, m.sed_emit, color='red',linestyle=':')
        for ij in range(0,int(m.parameters['nring'])):
            ax.loglog(m.sed_wave,m.sed_ringe[ij,:],linestyle='-',color='orange',alpha=0.1)
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
    ax.loglog(m.sed_wave, m.sed_star + m.sed_emit + m.sed_scat, color='black',linestyle='-')
    ax.errorbar(wav,flx,xerr=None,yerr=unc,linestyle='',marker='o',mec='black',mfc='white')
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

model = RTModel()

RTModel.get_parameters(model,'RTModel_Input_File.txt')

model.parameters['directory'] = '/Users/jonty/Desktop/'
model.parameters['prefix'] = 'test_2f_'
model.parameters['stype'] = 'blackbody'
model.parameters['tstar'] = 5950.0
model.parameters['rstar'] = 0.94
model.parameters['lstar'] = 1.318 #(4*np.pi*5.67e-8*(model.parameters['rstar']*rsol)**2*(model.parameters['tstar'])**4) / lsol
model.parameters['dstar'] = 48.0
model.parameters['mdust'] = 7.2e-3
model.parameters['q'] = -3.42
model.parameters['amin']  = 4.27
#model.parameters['rpeak'] = 100.
#model.parameters['dtype'] = 'gauss'
#model.parameters['rfwhm'] = 50.
model.parameters['dtype'] = 'onepl'
model.parameters['rin'] = 60.
model.parameters['rout'] = 140.
model.parameters['alpha_out'] = 0.0

print('here1')
RTModel.make_star(model)
print('here2')
RTModel.make_dust(model)
print('here3')
RTModel.make_disc(model)
print('here4')
RTModel.read_optical_constants(model)
print('here5')
RTModel.calculate_qabs(model)
print('here6')
RTModel.calculate_dust_emission(model,mode='full',tolerance=0.05)
print('here7')
RTModel.calculate_dust_scatter(model)
print('here8')
RTModel.flam_to_fnu(model)
print('here9')
make_sed(model)

end = time.time()
multi_time = end - start
print("SED calculations took: ",multi_time," seconds.")