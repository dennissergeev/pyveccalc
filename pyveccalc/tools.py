# -*- coding: utf-8 -*-
"""
Tools for manipulating data for use with `pyveccalc.standard.Wind*`
"""

import numpy as np

def __order_dims(d, inorder):
    if 'x' not in inorder or 'y' not in inorder:
        raise ValueError('a longitude-latitude (x-y) grid is required')
    lonpos = inorder.lower().find('x')
    latpos = inorder.lower().find('y')
    d = np.rollaxis(d, latpos)
    if lonpos < latpos:
        lonpos += 1
    d = np.rollaxis(d, lonpos)
    outorder = inorder.replace('x', '')
    outorder = outorder.replace('y', '')
    outorder = 'xy' + outorder
    return d, outorder


def __reshape(d):
    out = d.reshape(d.shape[:2] + (np.prod(d.shape[2:]),))
    return out, d.shape

def prep_data(data, dimorder):
    """
    Prepare data for input to `~pyveccalc.standard.Wind*`
    """
    # Returns the prepared data and some data info to help data recovery.
    pdata, intorder = __order_dims(data, dimorder)
    if len(pdata.shape) > 2:
        pdata, intshape = __reshape(pdata)
    else:
        # Do not add an extra dimension if array is 2D (e.g. lat/lon)
        intshape = pdata.shape
    info = dict(intermediate_shape=intshape,
                intermediate_order=intorder,
                original_order=dimorder)
    return pdata, info

def recover_data(pdata, info):
    """
    Recover the shape and dimension order of an array output from
    `~pyveccalc.standard.Wind*`
    """
    # Convert to intermediate shape (full dimensionality, windspharm order).
    data = pdata.reshape(info['intermediate_shape'])
    # Re-order dimensions correctly.
    rolldims = np.array([info['intermediate_order'].index(dim)
                         for dim in info['original_order'][::-1]])
    for i in range(len(rolldims)):
        # Roll the axis to the front.
        data = np.rollaxis(data, rolldims[i])
        rolldims = np.where(rolldims < rolldims[i], rolldims + 1, rolldims)
    return data
