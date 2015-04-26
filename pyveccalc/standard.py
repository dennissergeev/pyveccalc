# -*- coding: utf-8 -*-
"""
Wind vector calculations in finite differences
"""
#from __future__ import absolute_import
import numpy as np

class WindHorizontal(object):

    def __init__(self, u, v, x=1., y=1., hgridtype='regular'):
        """Initialize a VectorWind instance
        """
        self.hgridtype = hgridtype.lower()
        if u.shape != v.shape:
            raise ValueError('u and v must be the same shape')
        if len(u.shape) not in (2, 3, 4):
            raise ValueError('u and v must be rank 2, 3 or 4 arrays')
        if self.hgridtype not in ('regular', 'gaussian'):
            raise ValueError('invalid grid type: {0:s}'.format(repr(hgridtype)))
        
        self.u = u.copy()
        self.v = v.copy()
        self.x = x
        self.y = y
            
        nlon = u.shape[0]
        nlat = u.shape[1]

    def vort_z(self):
        """
        Relative vorticity
        """
        res = dfdx(self.v, self.x, 0) - dfdx(self.u, self.y, 1)
        return res       
        

def dfdx(f,x,axis=0):
    """
    Generic derivative
    """
    df = np.gradient(f)[axis]
    if type(x) is np.ndarray:
        dx = np.gradient(x)[axis]
    else:
        dx = x
    return df/dx
