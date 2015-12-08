# -*- coding: utf-8 -*-
"""
Physical parameters and functions for use with `pyveccalc.standard.Wind*`
"""

import numpy as np

#
# Physical parameters and constants for Earth's atmosphere
#
day_s    = 86400.         # Day length                              [s]
omg      = 2.*np.pi/day_s # Earth angular velocity                  [s^-1]
r_earth  = 6.371e6        # Earth's radius                          [m]

def calc_fcor(deg_lat):
    """Coriolis parameter for a given latitude in degrees"""
    return 2.*omg*np.sin(np.radians(deg_lat))

def lon2dist(lon, lat, R=r_earth):
    """
    Convert a longitude array to an array of distances (e.g. to use with numpy.gradient)

    **Inputs/Outputs**

    Variable   I/O   Description       Units   Default
    --------   ---   -----------       -----   -------
    lon         I    Longitude          deg    
    lat         I    Latitude           deg
    R           I    Planetary raduis    m     r_earth
    dist        O    Distance            m
    """
    dist = R*np.cos(np.radians(lat))*np.radians(lon)
    return dist 

def lat2dist(lat, R=r_earth):
    """
    Convert a latitude array to an array of distances (e.g. to use with numpy.gradient)

    **Inputs/Outputs**

    Variable   I/O   Description       Units   Default
    --------   ---   -----------       -----   -------
    lat         I    Latitude           deg
    R           I    Planetary raduis    m     r_earth
    dist        O    Distance            m
    """
    dist = R*np.radians(lat)
    return dist

def deg2name(num_deg, nbins=16):
    assert type(num_deg) != np.str, 'Input cannot be of string type'
    assert nbins==16 or nbins==8 or nbins==4, 'Number of bins must be 4, 8 or 16'
    db = 16/nbins
    deg_lev = np.linspace(0,360,nbins+1)
    deg_bound = deg_lev[1::] - deg_lev[1]/2.
    compass = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'][::db]

    if not hasattr(num_deg,'__iter__'):
        for j in xrange(len(deg_bound)):
            if deg_bound[j-1] < num_deg <= deg_bound[j]:
                out_deg = compass[j]
            if deg_bound[-1] < num_deg or num_deg <= deg_bound[0]:
                out_deg = compass[0]
                
    elif type(num_deg) is list:
        out_deg = []
        for i in num_deg:
            for j in xrange(len(deg_bound)-1):
                if deg_bound[j] < i <= deg_bound[j+1]:
                    out_deg.append(compass[j+1])
            if deg_bound[-1] < i or i <= deg_bound[0]:
                out_deg.append(compass[0])
            
    elif type(num_deg) is np.ndarray:
        out_deg = []
        for i in num_deg.flatten():
            for j in xrange(len(deg_bound)-1):
                if deg_bound[j] < i <= deg_bound[j+1]:
                    out_deg.append(compass[j+1])
            if deg_bound[-1] < i or i <= deg_bound[0]:
                out_deg.append(compass[0])
        out_deg = np.array(out_deg)
        out_deg = out_deg.reshape(num_deg.shape)
            
    else:
        raise TypeError('Handling input of type '+type(num_deg).__name__+' is not implemented yet')

    return out_deg
