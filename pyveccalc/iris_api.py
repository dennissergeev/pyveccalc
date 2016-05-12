# -*- coding: utf-8 -*-
"""
Iris interface to pyveccalc
"""
import numpy as np
from cached_property import cached_property
import cf_units
import iris
from iris.analysis.calculus import differentiate

from . import utils

def replace_dimcoord(cube, src_cube, axes='xy'):
    for axis in axes:
        oldcoord = cube.coord(axis=axis)
        newcoord = src_cube.coord(axis=axis)
        ndim = cube.coord_dims(oldcoord)[0]
        cube.remove_coord(oldcoord)
        cube.add_dim_coord(newcoord, ndim)


def replace_lonlat_dimcoord_with_cart(cube, dx=1, dy=None,
                                      rm_z_bounds=True,
                                      rm_z_varname=True,
                                      rm_surf_alt=True,
                                      rm_sigma=True,
                                      rm_aux_factories=True):
    res = cube.copy()
    if dy is None:
        dy = dx
    xpoints = np.array(range(cube.coord('grid_longitude').shape[0]))*dx
    ypoints = np.array(range(cube.coord('grid_latitude').shape[0]))*dy

    xcoord = iris.coords.DimCoord(points=xpoints, standard_name='longitude',
                                                  long_name='distance_along_x_axis',
                                                  units=cf_units.Unit('m'))
    ycoord = iris.coords.DimCoord(points=ypoints, standard_name='latitude',
                                                  long_name='distance_along_y_axis',
                                                  units=cf_units.Unit('m'))

    zcoord = iris.coords.DimCoord.from_coord(cube.coord('atmosphere_hybrid_height_coordinate'))
    res.remove_coord(cube.coord('atmosphere_hybrid_height_coordinate'))
    # Get rid of unnecessary coordinates that hinder cube.coord() method
    if rm_z_bounds:
        zcoord.bounds = None
    if rm_z_varname:
        zcoord.var_name = None
    if rm_surf_alt:
        res.remove_coord('surface_altitude')
    if rm_sigma:
        res.remove_coord('sigma')
    if rm_aux_factories:
        [res.remove_aux_factory(i) for i in res.aux_factories]

    [res.remove_coord(i) for i in res.dim_coords]

    res.add_dim_coord(zcoord,0)
    res.add_dim_coord(ycoord,1)
    res.add_dim_coord(xcoord,2)

    return res

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
    and regrid/interpolate the result from mid-points to the points of the input cube
    """
    res = differentiate(cube, coord)
    if res.coord(axis='x') != cube.coord(axis='x') or res.coord(axis='y') != cube.coord(axis='y'):
        res = res.regridded(cube)
    elif res.coord(axis='z') != cube.coord(axis='z'):
        cube_z_points = [(cube.coord(axis='z').name(), cube.coord(axis='z').points)]
        res = res.interpolate(cube_z_points, iris.analysis.Linear())
    return res

class AtmosFlow:
    """Atmospheric Flow in Cartesian coords"""
    def __init__(self, u, v, w, aux_cubes=None, lats=45.):
        self.u = u #.copy()
        self.v = v #.copy()
        self.w = w #.copy()

        self.main_cubes = list(filter(None, [u, v, w]))
        self.wind_cmpnt = list(filter(None, [u, v, w]))
        thecube = self.main_cubes[0]

        check_coords(self.main_cubes)

        # Get the dim_coord, or None if none exist, for the xyz dimensions
        self.xcoord = thecube.coord(axis='X')
        self.ycoord = thecube.coord(axis='Y')
        self.zcoord = thecube.coord(axis='Z')

        # TODO: correct fcor calculation
        self.fcor = utils.calc_fcor(lats)

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

    @cached_property
    def du_dx(self):
        r"""
        Derivative of u-wind along the x-axis
        .. math::
            u_x = \frac{\partial u}{\partial x}
        """
        return cube_deriv(self.u, self.xcoord)

    @cached_property
    def du_dy(self):
        r"""
        Derivative of u-wind along the y-axis
        .. math::
            u_y = \frac{\partial u}{\partial y}
        """
        return cube_deriv(self.u, self.ycoord)

    @cached_property
    def du_dz(self):
        r"""
        Derivative of u-wind along the z-axis
        .. math::
            u_z = \frac{\partial u}{\partial z}
        """
        return cube_deriv(self.u, self.zcoord)

    @cached_property
    def dv_dx(self):
        r"""
        Derivative of v-wind along the x-axis
        .. math::
            v_x = \frac{\partial v}{\partial x}
        """
        return cube_deriv(self.v, self.xcoord)

    @cached_property
    def dv_dy(self):
        r"""
        Derivative of v-wind along the y-axis
        .. math::
            v_y = \frac{\partial v}{\partial y}
        """
        return cube_deriv(self.v, self.ycoord)

    @cached_property
    def dv_dz(self):
        r"""
        Derivative of v-wind along the z-axis
        .. math::
            v_z = \frac{\partial v}{\partial z}
        """
        return cube_deriv(self.v, self.zcoord)

    @cached_property
    def dw_dx(self):
        r"""
        Derivative of w-wind along the x-axis
        .. math::
            w_x = \frac{\partial w}{\partial x}
        """
        return cube_deriv(self.w, self.xcoord)

    @cached_property
    def dw_dy(self):
        r"""
        Derivative of w-wind along the y-axis
        .. math::
            w_y = \frac{\partial w}{\partial y}
        """
        return cube_deriv(self.w, self.ycoord)

    @cached_property
    def dw_dz(self):
        r"""
        Derivative of w-wind along the z-axis
        .. math::
            w_z = \frac{\partial w}{\partial z}
        """
        return cube_deriv(self.w, self.zcoord)

    @cached_property
    def wspd(self):
        r"""
        Calculate wind speed (magnitude)
        .. math::
            \sqrt{u^2 + v^2 + w^2}
        """
        res = 0
        for cmpnt in self.wind_cmpnt:
            res += cmpnt**2
            res = res**0.5
            res.rename('wind_speed')
        return res

    @cached_property
    def rel_vort(self):
        r"""
        Calculate the vertical component of the vorticity vector
        .. math::
            \zeta = v_x - u_y
        """
        res = self.dv_dx - self.du_dy
        res.rename('atmosphere_relative_vorticity')
        return res

    @cached_property
    def div_h(self):
        r"""
        Calculate the horizontal divergence
        .. math::
            D_h = u_x + v_y
        """
        res = self.du_dx + self.dv_dy
        res.rename('divergence_of_wind')
        return res

    @cached_property
    def rel_vort_hadv(self):
        r"""
        Calculate the horizontal advection of relative vorticity
        .. math::
            \vec v\cdot \nabla \zeta
        """
        res = self.u*cube_deriv(self.rel_vort, self.xcoord) + \
              self.v*cube_deriv(self.rel_vort, self.ycoord)
        res.rename('horizontal_advection_of_atmosphere_relative_vorticity')
        return res

    @cached_property
    def rel_vort_vadv(self):
        r"""
        Calculate the vertical advection of relative vorticity
        .. math::
            w\frac{\partial \zeta}{\partial z}
        """
        res = self.w*cube_deriv(self.rel_vort, zcoord)
        res.rename('vertical_advection_of_atmosphere_relative_vorticity')
        return res

    @cached_property
    def rel_vort_stretch(self):
        r"""
        Stretching term
        .. math::
            \nabla\cdot\vec v (\zeta+f)
        """
        res = self.div_h*(self.rel_vort + self.fcor)
        res.rename('stretching_term_of_atmosphere_relative_vorticity_budget')
        return res

    @cached_property
    def rel_vort_tilt(self):
        r"""
        Tilting (twisting) term
        .. math::
            \vec k \cdot \nabla w\times\frac{\partial\vec v}{\partial z} =
            \frac{\partial w}{\partial x}*\frac{\partial v}{\partial z} -
            \frac{\partial w}{\partial y}*\frac{\partial u}{\partial z}
        """
        res = self.dw_dx * self.dv_dz - self.dw_dy * self.du_dz
        res.rename('tilting_term_of_atmosphere_relative_vorticity_budget')
        return res

    @cached_property
    def dfm_stretch(self):
        r"""
        Calculate the stretching deformation
        .. math::
            Def = u_x - v_y
        """
        res = self.du_dx - self.dv_dy
        res.rename('stretching_deformation_2d')
        return res

    @cached_property
    def dfm_shear(self):
        r"""
        Calculate the shearing deformation
        .. math::
            Def' = u_y + v_x
        """
        res = self.du_dy + self.dv_dx
        res.rename('shearing_deformation_2d')
        return res

    @cached_property
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

        Reference:
            http://dx.doi.org/10.3402/tellusa.v68.29464
        """
        numerator = self.rel_vort
        denominator = (self.div_h**2 + self.dfm_stretch**2 + self.dfm_shear**2)**0.5
        res = numerator/denominator
        res.rename('kinematic_vorticity_number_2d')
        return res
