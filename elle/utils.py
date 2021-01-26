#!/usr/bin/env python

from __future__ import (print_function, division)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.stats import norm
from scipy.optimize import curve_fit
from astropy.io import fits
import os

import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.ticker as plticker

def get_14_transit_duration(P, r_1, k, b, incl, ecc=0, omega=0):
    return (P / np.pi *
     np.arcsin(r_1 * np.sqrt((1 + k)**2 - b**2) / np.sin(incl)) *
     np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega))
    )

def get_23_transit_duration(P, r_1, k, b, incl, ecc=0, omega=0):
    return (P / np.pi *
     np.arcsin(r_1 * np.sqrt((1 - k)**2 - b**2) / np.sin(incl)) *
     np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega))
    )

def get_transit_mask(x, dur, t_exp=0, ref=0):
#    dur = (P / np.pi *
#     np.arcsin(r_1 * np.sqrt((1 + k)**2 - b**2) / np.sin(incl)) *
#     np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega))
#    ) / P

#    t_exp /= 60*60*24*P

    return ((x-ref) > -(dur+t_exp)/2) & ((x-ref) < (dur+t_exp)/2)

class Instrument:

    airmass  = None
    ccf      = None
    ccf_err  = None
    contrast = None
    fwhm     = None
    rv       = None
    rv_err   = None
    rv_start = None
    rv_step  = None
    seeing   = None
    snr50    = None
    texp     = None
    time     = None
    


class HARPS(Instrument):

    airmass  = "HIERARCH ESO TEL AIRM START"
    ccf      = 0
    contrast = "HIERARCH ESO DRS CCF CONTRAST"
    fwhm     = "HIERARCH ESO DRS CCF FWHM"
    rv       = "HIERARCH ESO DRS CCF RVC"
    rv_err   = "HIERARCH ESO DRS CCF NOISE"
    start    = "CRVAL1"
    step     = "CDELT1"
    seeing   = "HIERARCH ESO TEL AMBI FWHM START"
    snr50    = "HIERARCH ESO DRS SPE EXT SN50"
    texp     = "EXPTIME"
    time     = "HIERARCH ESO DRS BJD"

    labels = ['airmass',
              'contrast', 
              'fwhm', 
              'rv', 
              'rv_err',
              'seeing',
              'snr50',
              'texp',
              'time']

class CORALIE(Instrument):

    airmass  = "HIERARCH ESO OBS TARG AIRMASS"
    ccf      = 0
    contrast = "HIERARCH ESO DRS CCF CONTRAST"
    fwhm     = "HIERARCH ESO DRS CCF FWHM"
    rv       = "HIERARCH ESO DRS CCF RVC"
    rv_err   = "HIERARCH ESO DRS CCF NOISE"
    start    = "CRVAL1"
    step     = "CDELT1"
    snr50    = "HIERARCH ESO DRS SPE EXT SN50"
    texp     = "EXPTIME"
    time     = "HIERARCH ESO DRS BJD"

    labels = ['airmass',
              'contrast', 
              'fwhm', 
              'rv', 
              'rv_err',
              'snr50',
              'texp',
              'time']

class ESPRESSO(Instrument):
    raise NotImplementedError

def get_header(files, name):
    return np.array([fits.getval(x, name) for x in files])


def read(files,
        instrument='harps',
        oversample=1,
        order=-1):

    if instrument == 'harps':
        instr = HARPS()


        observables = np.recarray(
                (len(files),),
                [ (label, '<f8') for label in instr.labels ]
                )

    for label in instr.labels:
        # basically: observables.airmass = get_header(files, instr.airmass)
        value = get_header(files, 
                           getattr(instr, label)
                           )

        setattr(observables,
                label,
                value
                )

    N     = len(observables.time)
    start = get_header(files, instr.start)
    step  = get_header(files, instr.step)

    rv  = np.atleast_2d(
            [ start[i] + step[i] * np.arange(N) for i in range(N) ]
            )
    ccf = np.atleast_2d(
            [ fits.getdata(x, instr.ccf)[order,:] for x in files ]
            )

    return rv[:,::oversample], ccf[:,::oversample], observables


def read_fits(files, order=-1, instrument='coralie', oversample=1, bad=None):
    n = len(files)
    files.sort()

    m          = np.ones(n, dtype=bool)
    time       = np.zeros(n)
    texp       = np.zeros(n)
    fwhm       = np.zeros(n)
    rv_int     = np.zeros(n)
    rv_int_err = np.zeros(n)
    airmass    = np.zeros(n)
    contrast   = np.zeros(n)
    snr50      = np.zeros(n)
    ccf        = []
    rv         = []

    if instrument == 'coralie':
        bjd_hdr    = "HIERARCH ESO DRS BJD"
        texp_hdr   = "EXPTIME"
        fwhm_hdr   = "HIERARCH ESO DRS CCF FWHM"
        vref_hdr   = "CRVAL1"
        vstep_hdr  = "CDELT1"
        rv_hdr     = "HIERARCH ESO DRS CCF RVC"
        rv_err_hdr = "HIERARCH ESO DRS CCF NOISE"
        airmass_hdr = "HIERARCH ESO OBS TARG AIRMASS"
        contrast_hdr = "HIERARCH ESO DRS CCF CONTRAST"
        snr_hdr      = "HIERARCH ESO DRS SPE EXT SN50"
    elif instrument == 'harps-n':
        bjd_hdr    = "HIERARCH TNG DRS BJD"
        texp_hdr   = "EXPTIME"
        fwhm_hdr   = "HIERARCH TNG DRS CCF FWHM"
        vref_hdr   = "CRVAL1"
        vstep_hdr  = "CDELT1"
        rv_hdr     = "HIERARCH TNG DRS CCF RVC"
        rv_err_hdr = "HIERARCH TNG DRS CCF NOISE"

    
    for i in range(n):

        _ccf = fits.getdata(files[i], 0)[order, :]

        hdr = fits.getheader(files[i], 0)

        time[i]       = hdr[bjd_hdr]
        texp[i]       = hdr[texp_hdr]
        fwhm[i]       = hdr[fwhm_hdr]
        rv_int[i]     = hdr[rv_hdr]
        rv_int_err[i] = hdr[rv_err_hdr]
        airmass[i]    = hdr[airmass_hdr]
        contrast[i]    = hdr[contrast_hdr]
        snr50[i]      = hdr[snr_hdr]
        
        v0 = hdr[vref_hdr]
        dv = hdr[vstep_hdr]
        nv = _ccf.shape[0]

        ccf.append(_ccf)
        rv.append(np.arange(v0, v0+nv*dv, dv))

    sigma = fwhm / (2*np.sqrt(2 * np.log(2)))
    rv    = np.atleast_2d(rv)

#    print(rv)

    ccf   = np.atleast_2d(ccf)

    # check that all RV arrays start at the same reference, if not just get the
    # intersection
#    if not _all_equal(rv):
#        # get max and min RV from first and last columns
#        rv_min = np.max(rv[:,0])
#        rv_max = np.min(rv[:,-1])
#
#        # indices where to start and end
#        col_start = np.where(rv == rv_min)[1]
#        col_end   = np.where(rv == rv_max)[1]
#
#        rv_cut  = []
#        ccf_cut = []
#        for i in range(len(rv)):
#            rv_cut.append(rv[i][col_start[i]:col_end[i]+1])
#            ccf_cut.append(ccf[i][col_start[i]:col_end[i]+1])
#
#        rv  = np.atleast_1d(rv_cut[0])
#        ccf = np.atleast_2d(ccf_cut)
#    else:
#        rv = rv[0]

    if bad is not None:
        bad = np.atleast_1d(bad)
        m[bad] = False
    
    return (
        rv[m,::oversample],
        ccf[m,::oversample],
        time[m],
        rv_int[m],
        rv_int_err[m],
        texp[m],
        sigma[m],
        airmass[m],
        contrast[m],
        snr50[m])


def estimate_ccf_err(ccf, mask=None):
    
    if mask is None:
        m = np.ones_like(ccf, dtype=bool)
    else:
        m = mask
     
    # if return 2d array
#    ccf_err = np.ones_like(ccf)

#    error = np.sqrt(ccf)
    error = np.array(
            [np.std(ccf[i][~m[i]]) for i in range(ccf.shape[0])]
            )

    # error is 1d
    return error

#    ccf_err *= error[:, None]
    

    # if return 2d array
#    return ccf_err

def get_ccf_residuals(master_out, ccf, mask=None):
    if mask is None:
        mask = np.ones(ccf.shape[0], dtype=bool)

    return master_out - ccf[mask,:]


def create_master_out(rv, ccf, ccf_err, velocity_mask=None, bad=None):
    # stack the out of transit CCFs
#    tm = transit_mask
    
    vm = velocity_mask
    if vm is None:
        vm = np.ones_like(rv, dtype=bool)

    if bad is not None:
        keep = np.arange(ccf.shape[0])
        bad = np.atleast_1d(bad)
        keep = np.setdiff1d(keep, bad)

        ccf = ccf[keep]
        ccf_err = ccf_err[keep]
#        tm = tm[keep] 



#    rv_out      = np.median(rv[~m,:], axis=0)

#    assert _all_equal(rv), "input RV arrays are not equal, master RV array is ambiguous"   

#    ccf_out     = np.sum(ccf[~m], axis=0)
    ccf_out = np.average(ccf, weights=1/ccf_err**2, axis=0)

#    ccf_out = np.mean(ccf[~tm], axis=0)
    ccf_err_out = np.sqrt(np.sum(ccf_err**2, axis=0))

    # fit to shift to stellar rest frame and normalise
    popt, _ = fit_ccf(rv[vm], ccf_out[vm], yerr=ccf_err_out[vm])
    
    c, mu = popt[[0,2]]

    print('mu = {:.4f} +- {:.4f}'.format(mu, _[2]))#, mu)
#    print(what)
#    rv_out       = rv - mu
#    ccf_out     /= c
#    ccf_err_out /= c

    return (rv-mu, ccf_out, ccf_err_out, 
            inverted_normal_distribution(rv, *popt))# / c)


def inverted_normal_distribution(x, c, A, mu, sigma):
    # model function for the inverted Gaussian distribution
#    return c + A * norm.pdf(x, loc=mu, scale=sigma)
    return (c - 
            A * np.exp(
#                -0.5*(x - mu)**2 / (2*sigma**2)
                -0.5 * ( (x - mu) / sigma )**2
                )#/ 
#            (sigma * np.sqrt(2*np.pi))
            )

def fit_ccf_residuals_pymc3(rv, ccf, ccf_err=None, step=1000, tune=1000,
        chains=2, cores=2,
        target_accept=0.8):
    try:
        import pymc3 as pm
        import theano.tensor as tt
    except ImportError("pymc3 or theano not installed; either install or use "
                       "utils.fit_ccf_residuals()"):
        sys.exit()

#    varnames = ["c", "A", "mu", "sigma", "jitter", "mod"]

    with pm.Model() as model:

#        logA = pm.Normal('logA', mu=0, sd=10)
#        logc    = pm.Normal('logc', mu=0, sd=10)

#        c = pm.Normal('c', mu=0, sd=10)
#        A = pm.Bound(pm.Normal, upper=0)('A', mu=-0.1, sd=5)
#        jitter = pm.Bound(pm.Normal, lower=0)('jitter', mu=0.01, sd=5)
#        A = pm.Normal('A', mu=

#        c = pm.Normal('c', mu=0, sd=10)

#        c = pm.Normal('c', mu=0, sd=1)
        med = np.median(ccf)

#        c = pm.Normal('c', mu=med, sd=np.std(ccf))
        c_ppt = pm.Normal('c_ppt', mu=0, sd=np.std(ccf)*1e3)
        c     = pm.Deterministic('c', c_ppt/1e3)
#        c = pm.Normal('c', mu=0, sd=np.std(ccf)*1e3)
#        c = 0.0
#        c = pm.Uniform('c', -1, 1)

        rv_range = rv[-1] - rv[0]
#        rv_range = 20

#        mu = pm.Bound(
#                pm.Normal, lower=rv[0], upper=rv[-1]
##                pm.Normal, lower=-15, upper=15,
#                )(
#                        'mu', mu=0, sd=5.0
##                        testval=rv[np.argmin(ccf)]
#                        )
#        mu = -5.
        # standard use
#        mu = pm.Normal('mu', mu=0, sd=rv_range/2)

        mu = pm.Bound(pm.Normal, lower=-5, upper=5)(
                'mu', mu=0, sd=rv_range/2
                        )

#        sigma = pm.Bound(pm.Normal, lower=0, upper=rv_range/2)(
#                'sigma', mu=5, sd=rv_range)

#        sigma = pm.Bound(pm.HalfNormal, lower=0)('sigma', sd=10)
#        sigma = pm.Bound(pm.HalfNormal, lower=0)('sigma', sd=5)
#        sigma = pm.Bound(pm.HalfNormal, lower=0.5)('sigma', sd=2)
        sigma = pm.HalfNormal('sigma', sd=10)
#        sigma = 3
#        log_sigma = pm.Normal('log_sigma', mu=1.1, sd=0.5)
#        sigma = pm.Deterministic('sigma', tt.exp(log_sigma))

        fwhm = pm.Deterministic('fwhm', sigma * 2 * tt.sqrt(2*tt.log(2)))

#        sigma = pm.Uniform('sigma', 0, rv_range*3)

#        log_sigma = pm.Normal('log_sigma', mu=0, sd=np.log((rv[-1]-rv[0])*3))
#        log_sigma = pm.Normal('log_sigma', mu=0, sd=np.log(rv_range))
#        log_sigma = pm.Bound(pm.Normal, upper=rv_range*3)(
#                'log_sigma', mu=0, sd=10)


#        logjitter = pm.Normal('logs', mu=0, sd=10)
#        logjitter = pm.Normal('logs', mu=0, sd=10)

#        sigma = pm.Deterministic('sigma', tt.exp(log_sigma))
#        c = pm.Deterministic('c', tt.exp(logc))
#        A = pm.Deterministic('A', tt.exp(logA))

#        A = pm.Uniform('A', lower=0, upper=1)
#        A = pm.HalfNormal('A', sd=np.abs(ccf.max() - ccf.min()))
        A_ppt = pm.HalfNormal('A_ppt', sd=np.abs(ccf.max() - ccf.min()) * 1e3)
        A = pm.Deterministic('A', A_ppt/1e3)
#        A = 0.0002
#        A = pm.Bound(pm.Normal, lower=1e-6, upper=0.1)('A', mu=0.002, sd=0.01)
#        print(np.abs(
#                        np.min(ccf) - np.median(ccf)
#                        ))

#        logA = pm.Normal('logA',
#                mu=np.log(
#                    np.abs(
#                        np.min(ccf) - np.median(ccf)
#                        )
#                    ),
#                sd=5)
#        A    = pm.Deterministic('A', tt.exp(logA))
#        A = pm.Bound(pm.Normal, lower=0, upper=1)
#                'A', mu
#        jitter = pm.Uniform('jitter', 0, 1)
#        jitter = pm.Deterministic('jitter', tt.exp(logjitter))
    #     model = c - A * pm.Normal('')
        mod = (c -
#                A * tt.exp(-0.5*(rv - mu)**2 / sigma**2)#/ 
                A * tt.exp(-0.5 * tt.sqr((rv - mu) / sigma))#/ 
#                (sigma * np.sqrt(2*np.pi))
                )

        models = pm.Deterministic('models', mod)

        if ccf_err is None:
#        jitter = pm.Bound(pm.HalfNormal, lower=0)('jitter', sd=1)
            jitter_ppt = pm.HalfNormal('jitter_ppt', sd=np.std(ccf)*1e3)
            jitter = pm.Deterministic('jitter', jitter_ppt/1e3)
#            log_jitter = pm.Normal('log_jitter', mu=np.log(np.std(ccf)*10), sd=1)
#            jitter = pm.Deterministic('jitter', tt.exp(log_jitter))
#            jitter = pm.Bound(pm.HalfNormal, lower=0)('jitter', sd=np.std(ccf))
            obs = pm.Normal('obs', mu=mod, sd=jitter, observed=ccf)
        else:
            obs = pm.Normal('obs', mu=mod, sd=ccf_err, observed=ccf)

#        obs = pm.Normal('obs', mu=mod, sd=np.median(ccf_err), observed=ccf)

        trace = pm.sample(step, tune=tune, chains=chains, cores=cores,
                target_accept=target_accept)

    return trace


def fit_ccf_residuals(rv, ccf, err=None, p0=None, **kwargs):

    n      = ccf.shape[0]
#    mu     = np.zeros(n)
#    mu_err = np.zeros(n)
#    models = np.zeros_like(ccf)
#    (level, contrast, centre, width,
#     level_err, contrast_err, centre_err, width_err) = np.zeros((8,n))

    popts, perrs = [], []

    if err is None:
        err = np.ones_like(ccf)

    for i in range(n):

        popt, perr = fit_ccf(rv, ccf[i], yerr=err[i], p0=p0,
                            **kwargs)
#        for j,x in enumerate([level, contrast, centre, width]):
#            x[i] = popt[j]
#            level[i] = 
#        level[i] = popt[0]
        popts.append(popt)
        perrs.append(perr)
#        mu[i]      = popt[2]
#        mu_err[i]  = perr[2]

#        if len(popt) > 4:
#            s = slice(0,-1)
#        else:
#            s = slice(0,None)

#        models[i]  = inverted_normal_distribution(rv, *popt[s])

#    return mu, mu_err, models
    return np.atleast_2d(popts), np.atleast_2d(perrs)

def _log_probability(theta, x, y, yerr):

    def _log_likelihood(data, model, error):
        inv_sigma2 = 1/error**2
        return -0.5 * np.sum((data - model)**2 * inv_sigma2 - np.log(inv_sigma2))

    c, A, mu, sigma, jitter = theta

    if A > 0 or A < -10:
        return -np.inf
    elif jitter < 0:
        return -np.inf
    elif sigma < 0 or sigma > (x[-1]-x[0])/4:
        return -np.inf
    elif mu < x[0] or mu > x[-1]:
        return -np.inf
    elif c < -1 or c > 1:
        return -np.inf

    model = inverted_normal_distribution(x, c, A, mu, sigma)

    error = np.sqrt(yerr**2 + (model * jitter)**2)
    return _log_likelihood(y, model, error)


def fit_ccf(rv, ccf, yerr=None, p0=None, method='lsq', mcmc_steps=2000,
        mcmc_threads=2):
    x = rv
    y = ccf

    if yerr is None:
        yerr = None#np.ones_like(y)
        abs_sig = False
    else:
        abs_sig=True

    # initial guesses
#    if p0 is None:
    c0      = np.median(y)
#    A0      = np.abs(c0 / np.min(y))
    A0      = np.abs(c0 - np.min(y))
    mu0     = x[np.argmin(y)]
    sigma0  = 3

    default = [c0, A0, mu0, sigma0]

#        alpha0 = 0.0

    _p0 = []
    if isinstance(p0, list):
        for i in range(len(p0)):
            if p0[i] is None:
                _p0.append(default[i])
            else:
                _p0.append(p0[i])
        p0 = _p0
    elif p0 is None:
        p0 = default

    f = inverted_normal_distribution


#    try:
    if method == 'lsq':
        popt, pcov = curve_fit(f,
                               x, y, p0=p0, method='lm',
                               sigma=np.ones_like(y)*yerr,
                               absolute_sigma=True,
#                               sigma=yerr,
#                               absolute_sigma=abs_sig
                               )
        perr = np.sqrt(np.diag(pcov))

    elif method == 'mcmc':
        try:
            import emcee
            from multiprocessing import Pool
        except ImportError("emcee not installed; either install or use method='lsq'"):
            sys.exit()



        walkers = 400
        ndim    = 5


        p0.append(np.median(yerr)*0.01)
        p0_err = np.array([p0[0]*0.02, p0[1]*0.02, 0.5, p0[3]*0.02, p0[4]*0.1])
        p0     = np.array(p0)
        print(p0)
        print(p0_err)
        start  = [p0 + np.random.randn(ndim) * p0_err for _ in range(walkers)]
#        print(start)


        if mcmc_threads > 1:
            os.environ["OMP_NUM_THREADS"] = "1"
            with Pool(processes=2) as pool:
                sampler = emcee.EnsembleSampler(walkers, ndim,
                                                _log_probability,
                                                pool=pool,
                                                args=(x, y, yerr))
                sampler.run_mcmc(start, mcmc_steps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(walkers, ndim,
                                            _log_probability,
                                            args=(x, y, yerr))
            sampler.run_mcmc(start, mcmc_steps, progress=True)

        discard = int(0.75*mcmc_steps)
        fc = sampler.get_chain(flat=True, discard=discard)
        fcm = sampler.get_chain()
        print('fcm shape', fcm.shape)
        print('fc shape', fc.shape)
        steps = np.arange(mcmc_steps)

        fig, axes = plt.subplots(ndim+1,1, figsize=(10,2*ndim),
                gridspec_kw={"hspace":0.01})

        labels = ['logp', 'c', 'A', 'mu', 'sigma', 'jitter']
        for i in range(ndim+1):
            for j in range(walkers):
                if i == 0:
#                    print(sampler.get_log_prob().shape)
                    axes[i].plot(steps, sampler.get_log_prob()[:,j], lw=0.5)
                else:
                    axes[i].plot(steps, fcm[:,j,i-1], lw=0.5)
                axes[i].set_ylabel(labels[i])

                if i == 4:
                    axes[i].set_xlabel('steps')

        popt = np.median(fc, axis=0)
        perr = np.std(fc, axis=0)

#                                bounds=bounds,
#                                 xtol=1e-10, ftol=1e-10)
#    except ValueError:
#        plt.plot(x, y)
#        plt.show()

    return popt, perr

def get_continuum_mask(rv, ccf, sigma=None, fwhm=None, k=4):

    if sigma is None and fwhm is None:
        raise ValueError("Either sigma or fwhm need to be provided to mask "
                         "the line core")

    if fwhm is not None and sigma is None:
        fwhm  = np.atleast_1d(fwhm)
        sigma = fwhm / (2*np.sqrt(2 * np.log(2)))

    if sigma is not None and fwhm is None:
        sigma = np.atleast_1d(sigma)

    # find RV at CCF minimum
    rv0 = np.array([rv[i,np.argmin(ccf[i])] for i in range(ccf.shape[0])])
#    rv0 = rv[np.argmin(ccf, axis=1)]
#    print('rv0', rv0)

    m = np.array(
            [
                (rv[i] > (rv0[i] - k*sigma[i])) & (rv[i] < (rv0[i] + k*sigma[i]))
                for i in range(ccf.shape[0])
                ]
            )

    return m

def estimate_continuum(ccf, err=None, mask=None):#sigma=None, fhwm=None, k=3):

    m = mask
    if mask is None:
        m = np.ones_like(ccf)

    if err is None:
        err = np.ones_like(ccf)

    n = ccf.shape[0]

    # find RV at CCF minimum
#    rv0 = rv[np.argmin(ccf, axis=1)]
#    rv0 = np.array(
#            [
#                rv[np.argmin(ccf[i])] 
#                for i in range(n)
#                ]
#            )

    c = np.array(
            [
                np.average(
                    ccf[i][~m[i]], weights=1/err[i][~m[i]]**2
                    )
                for i in range(n)
                ]
            )

    return c

def normalise_ccf(ccf, normalisation, err=None):
    if err is None:
        return ccf * normalisation[:,None]
    else:
        return ccf * normalisation[:,None], err * normalisation[:,None]

#def normalise_by_light_curve(ccf, model, yerr=None):
#    if yerr is None:
#        return ccf * model[:,None]
#    else:
#        return ccf * model[:,None], yerr * model[:,None]

#def remove_keplerian_orbit(rv, ccf, rv_orb, ccf_err=None, method='cubic',
#                           fill_value=None):
def resample2(rv, ccf, rv_orb, new_grid=None, method='cubic',
        fill_value=None):

#    print(ccf)
    if new_grid is None:
        new_grid = rv.copy()

    if fill_value is None:
        fill_value = np.max(ccf, axis=1)

#    print(fill_value)
#    if isinstance(fill_value, 
#
#    elif isinstance(fill_value, float):

#    rv  = np.atleast_2d(rv)
#    ccf = np.atleast_2d(ccf)

#    _ccf = np.zeros_like(ccf)
    _ccf = []

    for i in range(ccf.shape[0]):
#        _ccf[i] = griddata(old_grid - rv_orb[i],
#                           ccf[i],
#                           new_grid,
#                           method=method),
        _ccf.append(
                griddata(rv[i] - rv_orb[i],
                         ccf[i],
                         new_grid[i],
                         method=method,
                         fill_value=fill_value[i]
                           )
                )

#                           fill_value=fill_value[i])
    _ccf = np.atleast_2d(_ccf)
#    print(new_grid.shape, _ccf.shape)

#    print(rv, _ccf, ccf_err)
    return _ccf



