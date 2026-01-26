# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:03:02 2021

@author: Robin
"""

import numpy as np
from scipy import interpolate
from astropy.io import ascii
from cmath import sqrt

silicate_file = 'astrosil.lnk'

sildata = ascii.read(silicate_file, comment='#')

astrosill = sildata["col1"].data
astrosiln = sildata["col2"].data
astrosilk = sildata["col3"].data

waterice_file = 'waterice.txt'

icedata = ascii.read(waterice_file, comment='#')

watericel = icedata["col1"].data
watericen = icedata["col2"].data
watericek = icedata["col3"].data

ice_frac = 0.2

fn = interpolate.interp1d(astrosill, astrosiln, fill_value="extrapolate")
fk = interpolate.interp1d(astrosill, astrosilk, fill_value="extrapolate")

mix_m = []


for idx, y in enumerate(watericel):
    m_sil = complex(fn(y), fk(y))
    m_ice = complex(watericen[idx], watericek[idx])
    b = 3*m_sil**2/(m_ice**2+2*m_sil**2)
    m_sq_eff = (((1-ice_frac)*m_sil**2 + 
                ice_frac*b*m_ice**2)/((1-ice_frac) + ice_frac*b))
    mix_m.append(sqrt(m_sq_eff))
    

final_l, final_n, final_k = watericel, [], []

for cnum in mix_m:
    final_n.append(cnum.real)
    final_k.append(cnum.imag)

with open('{:.0f}ice{:.0f}sil.lnk'.format(ice_frac*100, (1-ice_frac)*100), 'a') as file:
    for idx, y in enumerate(final_l):
        file.write('{:} {:} {:}\n'.format(y, final_n[idx], final_k[idx]))





