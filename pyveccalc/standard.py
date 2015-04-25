# -*- coding: utf-8 -*-
"""
Wind vector calculations in finite differences
"""
#from __future__ import absolute_import
import numpy as np

class VectorWind(object):

    def __init__(self, u, v, gridtype='regular'):
        """Initialize a VectorWind instance
        """
        # For both the input components check if there are missing values by
        # attempting to fill missing values with NaN and detect them. If the
        # inputs are not masked arrays then take copies and check for NaN.
        try:
            self.u = u.filled(fill_value=np.nan)
        except AttributeError:
            self.u = u.copy()
        try:
            self.v = v.filled(fill_value=np.nan)
        except AttributeError:
            self.v = v.copy()
        if np.isnan(self.u).any() or np.isnan(self.v).any():
            raise ValueError('u and v cannot contain missing values')
        # Make sure the shapes of the two components match.
        if u.shape != v.shape:
            raise ValueError('u and v must be the same shape')
        if len(u.shape) not in (2, 3):
            raise ValueError('u and v must be rank 2 or 3 arrays')
        nlat = u.shape[0]
        nlon = u.shape[1]
        try:
            # Create a Spharmt object to do the computations.
            self.gridtype = gridtype.lower()
            self.s = Spharmt(nlon, nlat, gridtype=self.gridtype)
        except ValueError:
            if self.gridtype not in ('regular', 'gaussian'):
                err = 'invalid grid type: {0:s}'.format(repr(gridtype))
            else:
                err = 'invalid input dimensions'
            raise ValueError(err)
