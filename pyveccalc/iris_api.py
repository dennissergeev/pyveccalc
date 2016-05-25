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

is_physical = lambda z: z.name() in ('height', 'level_height',
                                     'atmosphere_hybrid_height_coordinate',
                                     'pressure')
notdimless_and_1d = lambda z: not z.units.is_dimensionless() and z.ndim==1

iscube = lambda x: isinstance(x, iris.cube.Cube)
iscube_and_not_scalar = lambda x: isinstance(x, iris.cube.Cube) and x.shape != ()
def replace_dimcoord(cube, src_cube, axes='xy', return_copy=True):
    if return_copy:
        cp_cube = cube.copy()
    for axis in axes:
        oldcoord = cube.coord(axis=axis)
        newcoord = src_cube.coord(axis=axis)
        ndim = cube.coord_dims(oldcoord)[0]
        if return_copy:
            cp_cube.remove_coord(oldcoord)
            cp_cube.add_dim_coord(newcoord, ndim)
        else:
            cube.remove_coord(oldcoord)
            cube.add_dim_coord(newcoord, ndim)
    if return_copy:
        return cp_cube

def prepare_cube_zcoord(cube, rm_z_bounds=True, rm_z_varname=True):
    # Pick physically meaningful coordinate instead of level indices
    res = cube.copy()
    z_ax_coords = res.coords(axis='z')
    suitable_z = list(filter(notdimless_and_1d, z_ax_coords))
    if len(suitable_z) > 0:
        suitable_z = list(filter(is_physical, suitable_z))
        if len(suitable_z) == 0:
            # TODO: what if suitable_z > 1 ?
            msg = "Warning: the name of '{name}' is not among the preferred: {phys}"
            print(msg.format(name=z.name(), phys=', '.join(phys_coord)))
        zcoord = suitable_z[0]
    else:
        raise ValueError('No suitable coords among these: {z}'.format(z=z_ax_coords))

    if rm_z_bounds:
        zcoord.bounds = None
    if rm_z_varname:
        zcoord.var_name = None

    if not zcoord in res.dim_coords:
        for z in z_ax_coords:
            if z in res.dim_coords:
                zdim = res.coord_dims(z)[0]
                res.remove_coord(z)
        res.remove_coord(zcoord)
        zcoord = iris.coords.DimCoord.from_coord(zcoord)
        res.add_dim_coord(zcoord, zdim)
    return res

def replace_lonlat_dimcoord_with_cart(cube, dx=1, dy=None, dxdy_units='m'):
    res = cube.copy()
    if dy is None:
        dy = dx
    for axis, standard_name, step in zip(('x',         'y'),
                                         ('longitude', 'latitude'),
                                         ( dx,           dy)):
        icoord = res.coord(axis=axis)
        coord_dim = res.coord_dims(icoord)[0]
        res.remove_coord(icoord)

        eq_spaced_points = np.array(range(icoord.shape[0]))*step
        new_coord = iris.coords.DimCoord(points=eq_spaced_points,
                                         standard_name=standard_name,
                                         long_name='distance_along_{0}_axis'.format(axis),
                                         units=dxdy_units)

        res.add_dim_coord(new_coord, coord_dim)

    return res

def prepare_cube_on_model_levels(cube, lonlat2cart_kw={}, prep_zcoord_kw={},
                                 rm_surf_alt=True,
                                 rm_sigma=True,
                                 rm_aux_factories=True):
    res = cube.copy()
    if isinstance(lonlat2cart_kw, dict):
        res = replace_lonlat_dimcoord_with_cart(res, **lonlat2cart_kw)
    if isinstance(prep_zcoord_kw, dict):
        res = prepare_cube_zcoord(res, **prep_zcoord_kw)
    # Get rid of unnecessary coordinates that hinder cube.coord() method
    if rm_surf_alt:
        res.remove_coord('surface_altitude')
    if rm_sigma:
        res.remove_coord('sigma')
    if rm_aux_factories:
        # aka remove DerivedCoords
        [res.remove_aux_factory(i) for i in res.aux_factories]
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
    def __init__(self, u, v, w, lats=45., **kw_vars):
        self.u = u #.copy()
        self.v = v #.copy()
        self.w = w #.copy()

        self.__dict__.update(kw_vars)

        self.cubes = iris.cube.CubeList(filter(iscube, self.__dict__.values()))
        self.main_cubes = iris.cube.CubeList(filter(iscube_and_not_scalar, self.__dict__.values()))
        self.wind_cmpnt = iris.cube.CubeList(filter(None, [self.u, self.v, self.w]))
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
        msg += "\n".join(tuple(i.name() for i in self.cubes))
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
        Stretching deformation
        .. math::
            Def = u_x - v_y
        """
        res = self.du_dx - self.dv_dy
        res.rename('stretching_deformation_2d')
        return res

    @cached_property
    def dfm_shear(self):
        r"""
        Shearing deformation
        .. math::
            Def' = u_y + v_x
        """
        res = self.du_dy + self.dv_dx
        res.rename('shearing_deformation_2d')
        return res

    @cached_property
    def kvn(self):
        r"""
        Kinematic vorticity number

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

    @cached_property
    def density(self):
        try:
            # TODO: more flexible choosing
            #self.cubes.extract_strict('air_pressure')
            #self.cubes.extract_strict('air_potential_temperature')
            res = self.pres / (self.theta * (self.pres / self.p0) ** (self.R_d / self.c_p).data * self.R_d)
            res.rename('air_density')
            self.main_cubes(res)
            return res

        #except iris.exceptions.ConstraintMismatchError:
        except AttributeError:
            msg = 'The AtmosFlow has to contain cubes of pressure or potential temperature,\
                   as well as R_d, c_p and p0 constants to calculate air density'
            raise ValueError(msg)

    @cached_property
    def specific_volume(self):
        res = self.density ** (-1)
        res.rename('air_specific_volume')
        self.main_cubes(res)
        return res

    @cached_property
    def dp_dx(self):
        r"""
        Derivative of pressure along the x-axis
        .. math::
            p_x = \frac{\partial p}{\partial x}
        """
        return cube_deriv(self.pres, self.xcoord)

    @cached_property
    def dp_dy(self):
        r"""
        Derivative of pressure along the y-axis
        .. math::
            p_y = \frac{\partial p}{\partial y}
        """
        return cube_deriv(self.pres, self.ycoord)

    @cached_property
    def dsv_dx(self):
        r"""
        Derivative of specific volume along the x-axis
        .. math::
            \alpha_x = \frac{\partial\alpha}{\partial x}
        """
        return cube_deriv(self.specific_volume, self.xcoord)

    @cached_property
    def dsv_dy(self):
        r"""
        Derivative of specific volume along the y-axis
        .. math::
            \alpha_y = \frac{\partial\alpha}{\partial y}
        """
        return cube_deriv(self.specific_volume, self.ycoord)

    @cached_property
    def baroclinic_term(self):
        r"""
        Baroclinic term of relative vorticity budget

        .. math::
            \frac{\partial p}{\partial x}\frac{\partial\alpha}{\partial y}
            - \frac{\partial p}{\partial y}\frac{\partial\alpha}{\partial x}
        """
        res =   self.dp_dx * self.dsv_dy \
              - self.dp_dy * self.dsv_dx
        res.rename('baroclinic_term_of_atmosphere_relative_vorticity_budget')
        return res
