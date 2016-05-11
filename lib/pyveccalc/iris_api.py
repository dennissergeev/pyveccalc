# -*- coding: utf-8 -*-
"""
Iris interface to pyveccalc
"""
import numpy as np

import iris
from iris.analysis.calculus import differentiate

def check_coords(cubes):
    """Check the cubes coordinates for consistency"""
    # get the names of all coords binned into useful comparison groups
    coord_comparison = iris.analysis.coord_comparison(*cubes)

    bad_coords = coord_comparison['ungroupable_and_dimensioned']
    if bad_coords:
        raise ValueError("Coordinates found in one cube that describe "
                         "a data dimension which weren't in the other "
                         "cube ({}), try removing this coordinate.".format(
                             ', '.join(group.name() for group in bad_coords)))

    bad_coords = coord_comparison['resamplable']
    if bad_coords:
        raise ValueError('Some coordinates are different ({}), consider '
                         'resampling.'.format(
                             ', '.join(group.name() for group in bad_coords)))

def cube_deriv(cube, coord):
    """
    Wrapper to `iris.analysis.calculus.differentiate` to differentiate a cube
    and regrid the result to the points of the cube (not the mid-points)
    """
    cube_diff = differentiate(cube, coord)
    return cube_diff.regridded(cube)

class AtmosFlow:
    """Atmospheric Flow in Cartesian coords"""
    def __init__(self, u, v, w, aux_cubes=None):
        self.u = u
        self.v = v
        self.w = w
        
        self.main_cubes = list(filter(None, [u, v, w]))
        self.wind_cmpnt = list(filter(None, [u, v, w]))
        thecube = self.main_cubes[0]
        
        check_coords(self.main_cubes)

        # Get the dim_coord, or None if none exist, for the xyz dimensions
        self.xcoord = thecube.coord(axis='X')
        self.ycoord = thecube.coord(axis='Y')
        self.zcoord = thecube.coord(axis='Z')

        # Non-spherical coords?
        self.horiz_cs = thecube.coord_system('CoordSystem')
        self.spherical_coords = isinstance(self.horiz_cs, (iris.coord_systems.GeogCS,
                                                           iris.coord_systems.RotatedGeogCS))
        # interface for spherical coordinates switch?
        assert not self.spherical_coords, 'Only non-spherical coordinates are allowed so far...'
        
    def __repr__(self):
        msg = "pyveccalc 'Atmospheric Flow' containing of:\n"
        msg += "\n".join(tuple(i.name() for i in self.main_cubes))
        return msg
    
    @property
    def wspd(self):
        r"""
        Calculate wind speed (magnitude)
        
        .. math::
            \sqrt{u^2 + v^2 + w^2}
        """
        try:
            return self._wspd
        except AttributeError:
            res = 0
            for cmpnt in self.wind_cmpnt:
                res += cmpnt**2
            res = res**0.5
            res.rename('wind_speed')
            self._wspd = res
            return self._wspd
    
    @property
    def curl_z(self):
        r"""
        Calculate the vertical component of the vorticity vector
        .. math::
            \zeta = v_x - u_y
        """
        try: 
            return self._curl_z
        except AttributeError:
            dv_dx = cube_deriv(self.v, self.xcoord)
            du_dy = cube_deriv(self.u, self.ycoord)
            self._curl_z = dv_dx - du_dy
            self._curl_z.rename('atmosphere_relative_vorticity')
            return self._curl_z
      
    @property
    def div_h(self):
        r"""
        Calculate the horizontal divergence
        .. math::
            D_h = u_x + v_y
        """
        try:
            return self._div_h
        except AttributeError:
            du_dx = cube_deriv(self.u, self.xcoord)
            dv_dy = cube_deriv(self.v, self.ycoord)
            self._div_h = du_dx + dv_dy
            self._div_h.rename('divergence_of_wind')
            return self._div_h
    
    @property
    def dfm_stretch(self):
        r"""
        Calculate the stretching deformation
        .. math::
            Def = u_x - v_y
        """
        try:
            return self._dfm_stretch
        except AttributeError:
            du_dx = cube_deriv(self.u, self.xcoord)
            dv_dy = cube_deriv(self.v, self.ycoord)
            self._dfm_stretch = du_dx - dv_dy
            self._dfm_stretch.rename('stretching_deformation_2d')
            return self._dfm_stretch
    
    @property
    def dfm_shear(self):
        r"""
        Calculate the shearing deformation
        .. math::
            Def' = u_y + v_x
        """
        try:
            return self._dfm_shear
        except AttributeError:
            dv_dx = cube_deriv(self.v, self.xcoord)
            du_dy = cube_deriv(self.u, self.ycoord)
            self._dfm_shear = du_dy + dv_dx
            self._dfm_shear.rename('shearing_deformation_2d')
            return self._dfm_shear
    
    @property
    def kvn(self):
        r"""
        Calculate kinematic vorticity number
        
        .. math::
            W_k=\frac{||\Omega||}{||S||}=\frac{\sqrt{\zeta^2}}{\sqrt{D_h^2 + Def^2 + Def'^2}}
        where
        .. math::
            \zeta=v_x - u_y
            D_h = u_x + v_y
            Def = u_x - v_y
            Def' = u_y + v_x
        """
        try:
            return self._kvn
        except AttributeError:
            numerator = self.curl_z
            denominator = (self.div_h**2 + self.dfm_stretch**2 + self.dfm_shear**2)**0.5
            self._kvn = numerator/denominator
            self._kvn.rename('kinematic_vorticity_number_2d')
            return self._kvn