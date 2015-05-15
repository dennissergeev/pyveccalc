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
        """
        Initialize a WindHorizontal instance
        """
        if u.shape != v.shape:
            raise ValueError('u and v must be the same shape')
        if len(u.shape) not in (2, 3):
            raise ValueError('u and v must be rank 2, or 3 arrays')
        self.u = u.copy()
        self.v = v.copy()

        if len(u.shape) > 2:
            self.nd = u.shape[2]
        else:
            self.nd = 0

        self.hgridtype = hgridtype.lower()
        if self.hgridtype not in grids:
            raise ValueError('invalid grid type: {0:s}'.format(repr(hgridtype)))
        if self.hgridtype == 'lonlat':
            if type(x) is np.ndarray and type(y) is np.ndarray:
                if u.shape[:2] != x.shape or u.shape[:2] != y.shape:
                    if len(x.shape) == len(y.shape) == 1:
                        self.x, self.y = np.meshgrid(x, y)
                        self.x, self.y = self.x.T, self.y.T 
                        self.__lonlat2dist()
                        if u.shape[:2] != self.x.shape or u.shape[:2] != self.y.shape:
                            raise ValueError('Incorrect shape of coordinate arrays')
                    else:
                        raise ValueError('Incorrect shape of coordinate arrays')
                else:
                    self.x = x
                    self.y = y
                    self.__lonlat2dist()
            else:
                self.x = x
                self.y = y
        else:                
            self.x = x
            self.y = y

    def __lonlat2dist(self):
        """
        Converting input lon-lat arrays to distance arrays
        """
        self.x = utils.lon2dist(self.x, self.y)
        self.y = utils.lat2dist(self.y)

    def magnitude(self):
        """
        Calculate wind speed (magnitude of wind vector)
        """
        return np.sqrt(self.u**2 + self.v**2)

    def winddir_meteo(self, outfmt='numeric', nbins=16):
        """
        Calculate wind direction according to meteorological convention
        
        """
        f = 180.+180./np.pi*np.arctan2(self.u, self.v)
        if outfmt == 'name':
            f = utils.deg2name(f, nbins)
        return f 

    def vort_z(self):
        """
        Relative vorticity (z-component of curl)
        """
        f = dfdx(self.v, self.x, 0) - dfdx(self.u, self.y, 1)
       # if self.nd > 0:
       #     f = np.zeros(self.u.shape, dtype=self.u.dtype)
       #     for i in xrange(self.nd):
       #         f[:,:,i] = dfdx(self.v[:,:,i], self.x, 0) - dfdx(self.u[:,:,i], self.y, 1)
       # else:
       #     f = dfdx(self.v, self.x, 0) - dfdx(self.u, self.y, 1)
        return f
        

def dfdx(f,x,axis=0):
    """
    Generic derivative
    """
    df = np.gradient(f)[axis]
    if type(x) is np.ndarray:
        dx = np.gradient(x)[axis]
        if len(df.shape) == 3:
           dx = dx.reshape(dx.shape[:2] + (np.prod(dx.shape[2:]),)) 
    else:
        dx = x
    return df/dx


class ScalarHorizontal(object):

    def __init__(self, s, x=1., y=1., hgridtype=grids[0]):
        """
        Initialize a ScalarHorizontal instance
        """
        if len(s.shape) not in (2, 3):
            raise ValueError('Scalar s must be rank 2, or 3 arrays')
        self.s = s.copy()

        if len(s.shape) > 2:
            self.nd = s.shape[2]
        else:
            self.nd = 0

        self.hgridtype = hgridtype.lower()
        if self.hgridtype not in grids:
            raise ValueError('invalid grid type: {0:s}'.format(repr(hgridtype)))
        if self.hgridtype == 'lonlat':
            if type(x) is np.ndarray and type(y) is np.ndarray:
                if s.shape[:2] != x.shape or s.shape[:2] != y.shape:
                    if len(x.shape) == len(y.shape) == 1:
                        self.x, self.y = np.meshgrid(x, y)
                        self.x, self.y = self.x.T, self.y.T 
                        self.__lonlat2dist()
                        if s.shape[:2] != self.x.shape or s.shape[:2] != self.y.shape:
                            raise ValueError('Incorrect shape of coordinate arrays')
                    else:
                        raise ValueError('Incorrect shape of coordinate arrays')
                else:
                    self.x = x
                    self.y = y
                    self.__lonlat2dist()
            else:
                self.x = x
                self.y = y
        else:                
            self.x = x
            self.y = y

    def __lonlat2dist(self):
        """
        Converting input lon-lat arrays to distance arrays
        """
        self.x = utils.lon2dist(self.x, self.y)
        self.y = utils.lat2dist(self.y)

    def gradient(self):
        """
        Calculate horizontal components of gradient
        """
        fx = dfdx(self.s, self.x, 0)
        fy = dfdx(self.s, self.y, 1)
#        if self.nd > 0:
#            fx = np.zeros(self.s.shape, dtype=self.s.dtype)
#            fy = np.zeros(self.s.shape, dtype=self.s.dtype)
#            for i in xrange(self.nd):
#                fx[:,:,i] = dfdx(self.s[:,:,i], self.x, 0) 
#                fy[:,:,i] = dfdx(self.s[:,:,i], self.y, 1)
#        else:
#            fx = dfdx(self.s, self.x, 0)
#            fy = dfdx(self.s, self.y, 1)
        return fx, fy

    def gradient_mag(self):
        """
        Calculate magnitude of horizontal gradient
        """
        fx, fy = self.gradient()
        f = np.sqrt(fx**2 + fy**2)
        return f
