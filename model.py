#!/usr/bin/env python

from __future__ import (print_function, division)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

class ReloadedModel(object):
    def __init__(self, phase, r_1=0.01, r_p=0.1, i_p=90, Nxy=101, 
                 ld='quad', ldc=[0.2, 0.2],
                 oversample=1, dp=None):

        self.phase  = phase
        
        self.r_1    = r_1
        self.r_p    = r_p
        self.i_p    = np.deg2rad(i_p)
        self.ld     = ld
        self.ldc    = ldc
        self.Nxy    = Nxy
        self.os     = oversample
        self.dp     = dp
        
        # if oversample phase coverage
        if self.os > 1 and dp is not None:
            self.oversample = True
            self._phase_os = self._oversample_phase(self.phase)
        else:
            self.oversample = False
            self._phase_os = self.phase

        
        # create planet grid and all dependencies
        self._compute(self._phase_os)

    def __call__(self, v_eq, l, i_star, alpha, c=None, phase=None, r_1=None,
                 i_p=None):#, oversample=1, dp=None):

        recompute = False

        if r_1 is not None:
            self.r_1 = r_1
            recompute = True

        if i_p is not None:
            self.i_p = np.deg2rad(i_p)
            recompute = True

        if phase is None:
            phase = self.phase
        else:
            recompute = True

        if self.oversample:
            phase = self._oversample_phase(phase)

        if recompute:
            self._compute(phase)

        v_tot = self.v_stel_avg(self.v_stel(v_eq, np.deg2rad(l),
                                            np.deg2rad(i_star), alpha))
 
        if c is not None:
            v_tot += self.v_conv(*c)
        
        # average the oversampled points
        return np.array([np.mean(v_tot[i:i+self.os])
                         for i in xrange(0, len(v_tot), self.os)])

    def _oversample_phase(self, phase):
        return np.concatenate([np.linspace(p - self.dp/2, p + self.dp/2, self.os)
                               for p in phase]
                             )

    def _compute(self, phase):
        """ compute everything """
        self.compute_grid(phase)
        self.compute_mu()
        self.compute_intensity()
        self.compute_mu_avg()

    def compute_mu(self):
        """ computes mu on grid """
        # separation from grid points to star centre
        delta = np.sqrt(self.X_p**2 + self.Y_p**2)

        mu = np.sqrt(1 - delta**2)
        mu[np.isnan(mu)] = 0.0 # the ones outside stellar disc are nan: replace

        # mu for each grid point
        self._mu = mu


    def compute_occulted(self):

        X_pf, Y_pf = self.grid_pf
        X, Y       = self.grid_sf

        # select points inside planet disc
        inside_planet = np.sqrt(X**2 + Y**2) < self.r_p

        # select points inside stellar disc
        inside_star   = np.sqrt(X_pf**2 + Y_pf**2) < 1

        self._occulted = inside_star & inside_planet

    def compute_grid(self, phi):
        x_p = np.sin(2 * np.pi * phi) / self.r_1
        y_p = -np.cos(2 * np.pi * phi) * np.cos(self.i_p) / self.r_1

        x = np.linspace(-self.r_p, self.r_p, self.Nxy)
        y = np.linspace(-self.r_p, self.r_p, self.Nxy)

        # grid in stellar frame
        X, Y = np.meshgrid(x, y)

        # create copies of grid for each timestamp
        X = np.tile(X, (len(phi), 1, 1))
        Y = np.tile(Y, (len(phi), 1, 1))

        # bring grid to planet frame
        X_pf = X + x_p[:, np.newaxis, np.newaxis]
        Y_pf = Y + y_p[:, np.newaxis, np.newaxis]

        # set attributes
        self._grid_pf = [X_pf, Y_pf]
        self._grid_sf = [X, Y]
        self._x_p, self._y_p = x_p, y_p

        self.compute_occulted()

    def compute_brightness_average(self, y):
        """ computes the brightness-weighted average of y occulted by the planet"""
        assert len(y.shape) == 3 # check that input is 3d array

        return (np.sum((self.intensity * y) * self.occulted, axis=(1,2)) /
                np.sum(self.intensity * self.occulted, axis=(1,2)))

    def compute_intensity(self):
        """ computes the normalised intensity on grid according to the limb
        darkening law """
        self._intensity = self.limb_darkening_model(self.mu)

    def limb_darkening_model(self, mu):
        """ returns the intensity on mu given the chosen limb darkening law """
        return self.ld_quad(mu)

    def ld_quad(self, mu):
        """ returns intensity from a quadratic limb darkening law """
        return 1 - self.ldc[0]*(1 - mu) - self.ldc[1]*(1 - mu)**2

    def x_norm(self, l):
        return self.X_p * np.cos(l) -  self.Y_p * np.sin(l)

    def y_norm(self, l):
        return self.X_p * np.sin(l) + self.Y_p * np.cos(l)

    def z_norm(self, l):
        return np.sqrt(1 - self.x_norm(l)**2 - self.y_norm(l)**2)

    def z_norm_mark(self, l, i_star):
        beta = 0.5*np.pi - i_star
        return self.z_norm(l) * np.cos(beta) - self.y_norm(l) * np.sin(beta)

    def y_norm_mark(self, l, i_star):
        beta = 0.5*np.pi - i_star
        return self.z_norm(l) * np.sin(beta) + self.y_norm(l) * np.cos(beta)

    def v_stel(self, v_eq, l, i_star, alpha):
        """ returns the stellar rotational velocity at a given latitude """
        v = (self.x_norm(l) * v_eq * np.sin(i_star) * 
             (1 - alpha * self.y_norm_mark(l, i_star)**2))

        # replace nans with zeros
        v[np.isnan(v)] = 0.0

        return v

    def v_stel_avg(self, v_stel):
        """ returns the brightness-averaged stellar rotational velocity 
        occulted by the planet """
        # only select occulted mu's
        return self.compute_brightness_average(v_stel)

    def v_conv(self, *c):
        """ returns the convective contribution to the velocity """

        def _integrand(mu, i):
            """ function to integrate """
            return self.limb_darkening_model(mu) * mu**(i+1)

        c    = [0] + list(c)
        clen = len(c)
#        integral = np.array([quad(_integrand, a=0, b=1, args=(i), epsabs=1e-3)[0]
        integral = np.array([quad(_integrand, a=0, b=1, args=(i))[0]
                             for i in xrange(clen)])

        # check for nans
        assert ~np.any(np.isnan(integral)), "nan in integral"

        # don't quite understand why this is needed. MCMC could surely fit for
        # it?
#        if clen > 1:
        c0 = -np.sum(c[1:] * integral[1:]) / integral[0]
        c[0] = c0


        v = np.sum([c[i] * self.mu_avg**i for i in xrange(clen)], axis=0)

        return v


    def get_latitudes(self, l, i_star):
        """ returns the latitudes relative to the equator transited by the
            planets """
        l, i_star   = map(np.deg2rad, (l, i_star))
        ci          = int((self.Nxy - 1) / 2)
        positions   = self.y_norm_mark(l, i_star)[:, ci, ci]

        return np.rad2deg(np.arcsin(positions))


    def compute_mu_avg(self):
        """ numerically computes the brightness-weighted mu on the stellar grid """
        mu_avg = self.compute_brightness_average(self.mu)
        self._mu_avg = mu_avg

    @property
    def intensity(self):
        return self._intensity

    @property
    def mu(self):
        return self._mu

    @property
    def mu_avg(self):
        """ returns the brightness-weighted position on stellar disc, < mu > """
        return self._mu_avg

    @property
    def grid_sf(self):
        """ returns the grid in the stellar frame """
        return self._grid_sf

    @property
    def grid_pf(self):
        """ returns the grid in the planet frame """
        return self._grid_pf

    @property
    def occulted(self):
        """ returns booleans of occulted points in the grid """
        return self._occulted

    @property
    def X_p(self):
        return self.grid_pf[0]

    @property
    def Y_p(self):
        return self.grid_pf[1]

    @property
    def x_p(self):
        """ returns the planet x position """
        return self._x_p

    @property
    def y_p(self):
        """ returns the planet y position """
        return self._y_p

if __name__ == "__main__":

    phase = np.loadtxt('/Users/vxh710/PhD/software/reloaded/results/pimen/run2/191009/2018-12-16/all_remove_kepler_bin_disc_int/20.25min/fitted_rv.txt',
                       unpack=True, usecols=(0,))

    v_eq = 6.0
    l = -20.
    alpha = 0.6
    i_star = 120.

    phase_f = np.linspace(phase[0]-0.003, phase[-1]+0.003, 100)

    reloaded_kwargs = {'r_1':0.074449, 'i_p':np.rad2deg(1.52685),
            'r_p':np.sqrt(0.00028), 'ld':'quad', 'ldc':[0.28, 0.27]}

    relo1 = ReloadedModel(phase, **reloaded_kwargs)
    v_tot1 = relo1(v_eq, l, i_star, alpha, phase=phase_f)


    reloaded_kwargs = {'r_1':0.074449, 'i_p':np.rad2deg(1.52685),
            'r_p':np.sqrt(0.00028), 'ld':'quad', 'ldc':[0.28, 0.27],
            'oversample':5, 'dp':20.25/60/24/6.2682}
    relo5 = ReloadedModel(phase, **reloaded_kwargs)
    print(relo5.phase)
    print(relo5._phase_os)

    v_tot5 = relo5(v_eq, l, i_star, alpha, phase=phase_f)

    plt.figure()

    plt.plot(phase_f, v_tot1)
    plt.plot(phase_f, v_tot5)

    plt.show()

