# -*- coding: utf-8 -*-
"""
Compute magnitude and relative vorticity
"""

import numpy as np
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from netCDF4 import Dataset

from pyveccalc.standard import WindHorizontal
from pyveccalc.tools import prep_data, recover_data

example_data_path = './example_data/data.nc'

with Dataset(example_data_path) as f:
    lon = f.variables['longitude'][:]
    lat = f.variables['latitude'][:]
    u = f.variables['u'][:]
    v = f.variables['v'][:]
    vo = f.variables['vo'][:]

uu, info = prep_data(u,'tzyx')
vv, _ = prep_data(v,'tzyx')

W = WindHorizontal(uu,vv,lon,lat,'lonlat')

rel_vort = W.vort_z()

rel_vort = recover_data(rel_vort, info)

# Pick a level and a moment in time
# add a cyclic point (for plotting purposes)
rel_vort_c, lon_c = addcyclic(rel_vort[0,0,:,:], lon)
vo_c, _ = addcyclic(vo[0,0,:,:], lon)

#
# Plot the calculated relative vorticity along with relative vorticity from the dataset
# 
fig, (ax1, ax2) = plt.subplots(nrows=2)

# Note: excluding high latitudes due to small grid step and large errors in finite differences
m = Basemap(ax=ax1, projection='cyl', resolution='c', \
            llcrnrlon=0, llcrnrlat=-88, \
            urcrnrlon=360.01, urcrnrlat=88)
x, y = m(*np.meshgrid(lon_c, lat))
c1 = m.contourf(x, y, vo_c*1e4, cmap=plt.cm.RdBu_r,
               extend='both')
m.drawcoastlines()
m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0])
m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1])
plt.colorbar(c1, ax=ax1, orientation='horizontal')
ax1.set_title('Relative vorticity ($10^{-4}$s$^{-1}$)\n ERA-Interim', fontsize=16)

m = Basemap(ax=ax2, projection='cyl', resolution='c', \
            llcrnrlon=0, llcrnrlat=-88, \
            urcrnrlon=360.01, urcrnrlat=88)
x, y = m(*np.meshgrid(lon_c, lat))
c2 = m.contourf(x, y, rel_vort_c*1e4, cmap=plt.cm.RdBu_r,
               extend='both')
m.drawcoastlines()
m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0])
m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1])
plt.colorbar(c2, ax=ax2, orientation='horizontal')
ax2.set_title('Relative vorticity ($10^{-4}$s$^{-1}$)\n calculated by pyveccalc', fontsize=16)

fig.tight_layout()
plt.show()
