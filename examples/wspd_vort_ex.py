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

W = WindHorizontal(uu,vv,lons,lats,'lonlat')

rel_vort = W.vort_z()
wspd = W.magnitude()

rel_vort = recover_data(rel_vort, info)
wspd = recover_data(wspd, info)

# Pick a level and a moment in time
# add a cyclic point (for plotting purposes)
rel_vort_c, lons_c = addcyclic(rel_vort[0,0,:,:], lons)
wspd_c, _ = addcyclic(wspd[0,0,:,:], lons)

#
# Plot wind speed and relative vorticity
#
fig, (ax1, ax2) = plt.subplots(nrows=2)

m = Basemap(ax=ax1, projection='moll', resolution='c', \
            llcrnrlon=0, llcrnrlat=-90, \
            urcrnrlon=360.01, urcrnrlat=90)
x, y = m(*np.meshgrid(lons_c, lats))
c = m.contourf(x, y, wspd_c, cmap=plt.cm.hot_r,
               extend='both')
m.drawcoastlines()
m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0])
m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1])
plt.colorbar(c, ax=ax1, orientation='horizontal')
ax1.title('Wind speed (ms$^{-1}$)', fontsize=16)

m = Basemap(ax=ax2, projection='moll', resolution='c', \
            llcrnrlon=0, llcrnrlat=-90, \
            urcrnrlon=360.01, urcrnrlat=90)
x, y = m(*np.meshgrid(lons_c, lats))
c = m.contourf(x, y, rel_vort_c*1e4, cmap=plt.cm.RdBu_r,
               extend='both')
m.drawcoastlines()
m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0])
m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1])
plt.colorbar(c, ax=ax1, orientation='horizontal')
ax1.title('Relative vorticity ($10^{-4}$s$^{-1}$)', fontsize=16)

fig.tight_layout()
fig.show()
