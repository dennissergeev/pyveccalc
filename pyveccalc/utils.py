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

