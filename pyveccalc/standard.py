# -*- coding: utf-8 -*-
"""
Wind vector calculations in finite differences
"""
#from __future__ import absolute_import
import numpy as np
import utils

grids = ('descartes','lonlat') #,'gaussian')

class WindHorizontal(object):

    def __init__(self, u, v, x=1., y=1., hgridtype=grids[0]):
        """Initialize a VectorWind instance
        """
        if u.shape != v.shape:
            raise ValueError('u and v must be the same shape')
        if len(u.shape) not in (2, 3, 4):
            raise ValueError('u and v must be rank 2, 3 or 4 arrays')
        self.u = u.copy()
        self.v = v.copy()

        self.hgridtype = hgridtype.lower()
        if self.hgridtype not in grids:
            raise ValueError('invalid grid type: {0:s}'.format(repr(hgridtype)))
        if self.hgridtype == 'lonlat':
            if type(x) is np.ndarray and type(y) is np.ndarray:
               if u.shape[:2] != x.shape or u.shape[:2] != y.shape:
                  if len(x.shape) == len(y.shape) == 1:
                      self.x, self.y = np.meshgrid(x, y)
                      self.x, self.y = self.x.T, self.y.T 
                      if u.shape[:2] != self.x.shape or u.shape[:2] != self.y.shape:
                          raise ValueError('Incorrect shape of coordinate arrays')
                  else:
                      raise ValueError('Incorrect shape of coordinate arrays')
               else:
                   # Converting input lon-lat arrays to distance arrays
                   self.x = utils.lon2dist(self.x, self.y)
                   self.y = utils.lon2dist(self.y)
        else:                
            self.x = x
            self.y = y
        
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
