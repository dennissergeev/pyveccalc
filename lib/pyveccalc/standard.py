# -*- coding: utf-8 -*-
"""
Wind vector calculations in finite differences
"""
import numpy as np

from . import utils

grids = ('cartesian','lonlat') #,'gaussian')

# TODO: laplacian, see "High Performance Python", p. 118
#
class Wind3D(object):

    def __init__(self, u, v, w, x=1., y=1., z=1., lats=45., hgridtype=grids[0]):
        """
        Initialize a Wind3D instance
        """
        if (u.shape != v.shape) or (u.shape != w.shape):
            raise ValueError('u, v and w must be the same shape')
        #if len(u.shape) not in (2, 3):
        #    raise ValueError('u and v must be rank 2, or 3 arrays')
        self.u = u.copy()
        self.v = v.copy()
        self.w = w.copy()
        if isinstance(lats, np.ndarray):
            assert lats.shape == self.u.shape[:2], 'Reshape lats!'
        self.lats = lats

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

        if isinstance(z, np.ndarray):
            if z.shape[0] == u.shape[-1]:
                # Assuming z is an array of vertical levels
                self.z = z.reshape((1,)*2+z.shape)
            else:
                msg = 'z.shape={0} with u.shape={1} is not allowed'.format(z.shape, u.shape)
                raise ValueError(msg)
        else:
            # Assuming z is a scalar
            self.z = z


    def __lonlat2dist(self):
        """
        Converting input lon-lat arrays to distance arrays
        """
        self.x = utils.lon2dist(self.x, self.y)
        self.y = utils.lat2dist(self.y)

    def __assert_vort(self):
        if not hasattr(self, 'vo'): self.vo = self.vort_z()

    def vort_z(self):
        """
        Relative vorticity (z-component of curl)

        $\frac{\partial \zeta}{\partial t}$
        """
        f = dfdx(self.v, self.x, 0) - dfdx(self.u, self.y, 1)
        return f

    def hdiv(self):
        """
        Horizontal divergence

        $\nabla_p\cdot\vec v$
        """
        f = dfdx(self.u, self.x, 0) + dfdx(self.v, self.y, 1)
        return f

    def vort_tend_hadv(self):
        """
        Horizontal advection of relative vorticity

        $\vec v\cdot \nabla_p \zeta$
        """
        self.__assert_vort()
        f = self.u*dfdx(self.vo, self.x, 0) + \
            self.v*dfdx(self.vo, self.y, 1)
        f = -f
        return f

    def vort_tend_hadv_flux(self):
        """
        Horizontal advection in flux form

        $\nabla_p \cdot (\zeta\vec v)$
        """
        self.__assert_vort()
        f = dfdx(self.u*self.vo, self.x, 0) + \
            dfdx(self.v*self.vo, self.y, 1)
        f = -f
        return f

    def vort_tend_vadv(self):
        """
        Vertical advection of relative vorticity

        $\omega \frac{\partial \zeta}{\partial p}$
        """
        self.__assert_vort()
        f = - self.w*dfdx(self.vo, self.z, 2)
        return f

    def planet_vort_adv(self):
        """
        Planetary vorticity advection

        $\beta v$
        """
        fcor = utils.calc_fcor(self.lats)
        beta = dfdx(fcor,self.y,1)
        beta = beta.repeat(self.v.shape[-1]).reshape(self.v.shape)
        f = - self.v*beta
        return f

    def vort_tend_stretch(self):
        """
        Stretching term

        $\nabla_p\cdot\vec v (\zeta+f)$
        """
        self.__assert_vort()
        div = self.hdiv()
        fcor = utils.calc_fcor(self.lats)
        fcor = fcor.repeat(self.vo.shape[-1]).reshape(self.vo.shape)
        f = - div*(self.vo + fcor)
        return f

    def vort_tend_div_fcor(self):
        """
        Product of divergence and Coriolis parameter

        $f\nabla_p\cdot\vec v$
        """
        div = self.hdiv()
        fcor = utils.calc_fcor(self.lats)
        fcor = fcor.repeat(div.shape[-1]).reshape(div.shape)
        f = - div*fcor
        return f

    def vort_tend_twist(self):
        """
        Tilting/twisting term

        $\vec k \cdot \nabla\omega\times\frac{\partial\vec v}{\partial p}$
        """
        dwdx = dfdx(self.w, self.x, 0)
        dwdy = dfdx(self.w, self.y, 1)
        dudp = dfdx(self.u, self.z, 2)
        dvdp = dfdx(self.v, self.z, 2)
        f = - (dwdx*dvdp - dwdy*dudp)
        return f

    def vort_tend_rhs(self, form='standard'):
        """
        Right-hand side of vorticity equation in pressure coordinates

        Kwargs:
        -------
            form: str, standard | flux
                 standard : standard form
                 flux : horizontal advection in flux form, stretching term -> div*fcor
        Returns:
        --------
            sequence of terms of `numpy.ndarray` type

        Reference: Bluestein, 1992 Vol I, sect. 4.5.4
        """
        if form.lower() == 'standard':
            self.vo = self.vort_z()
            hadv = self.vort_tend_hadv()
            vadv = self.vort_tend_vadv()
            planet_vort_adv = self.planet_vort_adv()
            stretch = self.vort_tend_stretch()
            twist = self.vort_tend_twist()
        elif form.lower() == 'flux':
            self.vo = self.vort_z()
            hadv = self.vort_tend_hadv_flux()
            vadv = self.vort_tend_vadv()
            planet_vort_adv = self.planet_vort_adv()
            stretch = self.vort_tend_div_fcor()
            twist = self.vort_tend_twist()
        else:
            raise ValueError('`form` keyword should be either "standard" or "flux"')

        return hadv, vadv, planet_vort_adv, stretch, twist



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
       #     for i in range(self.nd):
       #         f[:,:,i] = dfdx(self.v[:,:,i], self.x, 0) - dfdx(self.u[:,:,i], self.y, 1)
       # else:
       #     f = dfdx(self.v, self.x, 0) - dfdx(self.u, self.y, 1)
        return f


def dfdx(f,x,axis=0):
    """
    Generic derivative
    """
    df = np.gradient(f)[axis]
    if isinstance(x, np.ndarray):
        #if len(df.shape) == 3: CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #   dx = dx.reshape(dx.shape[:2] + (np.prod(dx.shape[2:]),))
        if x.shape == df.shape:
            dx = np.gradient(x)[axis]
        elif x.shape[-1] == df.shape[-1]:
            dx = np.gradient(x.squeeze())
            dx = dx.reshape(x.shape)
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
#            for i in range(self.nd):
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
