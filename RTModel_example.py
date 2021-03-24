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

#benchmarking with time
start = time.time()

model = RTModel()

RTModel.get_parameters(model,'RTModel_Input_File.txt')

RTModel.make_star(model)

RTModel.make_dust(model)

RTModel.make_disc(model)

RTModel.read_optical_constants(model)

sed_tot = np.zeros(model.sed_wave.shape)
sed_ring = np.zeros((int(model.parameters['nring']),int(model.parameters['nwav'])))
sed_wav = model.sed_wave

dstar = model.parameters["dstar"]

#loop over grain size and radius to calculate dust emission
#calculate optical constants
# x = 2.*np.pi*ag/wav
# dust_nk = np.zeros(wav.shape,dtype='complex')
# qabs = np.zeros(wav.shape)
# for i in range(0,len(wav)):
#     dust_nk[i] = complex(dust_n[i],dust_k[i])
#     qext, qsca, qback, g = mpy.mie(dust_nk,x)
#     qabs[i] = (qext - qsca)
for ii in range(0,int(model.parameters['ngrain'])):  
    x = 2.*np.pi*model.ag[ii]/model.sed_wave
    qabs = np.zeros(model.sed_wave.shape)
    qext, qsca, qback, g = mpy.mie(model.oc_nk,x)
    qabs = (qext - qsca)
    
    for ij in range(0,int(model.parameters['nring'])):    
        scalefactor = model.ng[ii]*model.scale[ij]*((model.ag[ii]*um)**2)/(model.parameters['dstar']*pc)**2
        tdust = RTModel.calculate_dust_temperature(model,model.sed_star,model.ag[ii],qabs,model.radii[ij],blackbody=False,tolerance=0.01)        
        sed_flx  = scalefactor * qabs * np.pi * RTModel.planck_lam(model.sed_wave*um, tdust)
        sed_ring[ij,:] += sed_flx
        
        model.sed_disc += sed_flx  
        
#convert model fluxes from flam to fnu (in mJy) 
convertfactor = 1e3*1e26*(model.sed_wave*um)**2 /c

model.sed_rings = sed_ring*convertfactor
model.sed_disc  = model.sed_disc*convertfactor
model.sed_star  = model.sed_star*convertfactor
model.sed_total = (model.sed_star + model.sed_disc)

RTModel.make_sed(model)

end = time.time()
multi_time = end - start
print("SED calculations took: ",multi_time," seconds.")