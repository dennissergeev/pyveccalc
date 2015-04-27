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

uu, info = prep_data(u,'tzyx')
vv, _ = prep_data(v,'tzyx')

W = WindHorizontal(uu,vv,lon,lat,'lonlat')

wspd = W.magnitude()

wspd = recover_data(wspd, info)

# Pick a level and a moment in time
# add a cyclic point (for plotting purposes)
wspd_c, lon_c = addcyclic(wspd[0,0,:,:], lon)

#
# Plot wind speed and relative vorticity
#
fig, ax1 = plt.subplots()

m = Basemap(ax=ax1, projection='cyl', resolution='c', \
            llcrnrlon=0, llcrnrlat=-90, \
            urcrnrlon=360.01, urcrnrlat=90)
x, y = m(*np.meshgrid(lon_c, lat))
c1 = m.contourf(x, y, wspd_c, cmap=plt.cm.hot_r,
               extend='both')
m.drawcoastlines()
m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0])
m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1])
plt.colorbar(c1, ax=ax1, orientation='vertical')
ax1.set_title('Wind speed (ms$^{-1}$)', fontsize=16)

plt.show()
