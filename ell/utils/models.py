#!/usr/bin/env python

import numpy as np
import astropy.units as u
import sys

def get_light_curve(x, period, t0, aor, ror, incl,
                    ustar=None, ld=None, ecc=0, omega=90, sbratio=0,
                    oversample=1, texp=None, **kwargs):
    try:
        from ellc import lc
    except ModuleNotFoundError:
        print("Please install `ellc` to use this function")
        sys.exit()

    if not isinstance(x, np.ndarray):
        x = np.array([x])
    
    r1oa = 1/aor
    r2oa = r1oa * ror
    f_s = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
    f_c = np.sqrt(ecc) * np.cos(np.deg2rad(omega))

    model = lc(x,
            radius_1=r1oa,
            radius_2=r2oa,
            incl=incl,
            sbratio=sbratio,
            period=period,
            t_zero=t0,
            f_s=f_s,
            f_c=f_c,
            light_3=0, 
            ld_1=ld,
            ldc_1=ustar,
            t_exp=texp,
            n_int=oversample,
            **kwargs)
#            grid_1='very_sparse',
#            grid_2='v)

    return model

def get_radial_velocity(x, period, t0, incl, K, K2=None, ecc=0, omega=90, sbratio=0):
    try:
        from ellc import rv
    except ModuleNotFoundError:
        print("Please install `ellc` to use this function")
        sys.exit()


    K      *= u.m/u.s
    period *= u.day

    a1 = (K * period * np.sqrt(1 - ecc**2) /
            (2*np.pi * np.sin(np.deg2rad(incl)))
            ).to(u.R_sun).value

    if K2 is not None:
        K2 *= u.m / u.s
        q   = K / K2
        a2  = (K2 * period * np.sqrt(1 - ecc**2) /
                 (2*np.pi * np.sin(np.deg2rad(incl)))
                 ).to(u.R_sun).value
        a   = a1 + a2
    else:
        q   = 1
        a   = a1 * (1 + 1/q)

    fs = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
    fc = np.sqrt(ecc) * np.cos(np.deg2rad(omega))


    model = rv(x, 
            period=period.value, 
            t_zero=t0, 
            incl=incl, 
            sbratio=0,
            a=a, 
            q=q, 
            f_s=fs, 
            f_c=fc, 
            flux_weighted=False
            )

    if K2 is None:
        return model[0] * 1e3
    else:
        return [m * 1e3 for m in model]


def get_rossiter_mclaughlin(x, period, t0, aor, ror, incl, K, vsini, ell,
        ecc=0, omega=90, sbratio=0, ustar=None, ld=None,
        oversample=1, texp=None, slope=False
        ):

    try:
        from ellc import rv
    except ModuleNotFoundError:
        print("Please install `ellc` to use this function")
        sys.exit()

    K      *= u.m/u.s
    period *= u.day

    a1 = (K * period * np.sqrt(1 - ecc**2) /
            (2*np.pi * np.sin(np.deg2rad(incl)))
            ).to(u.R_sun).value
    q  = 1
    a  = a1 * (1 + 1/q)
    r1oa = 1/aor
    r2oa = r1oa * ror
    f_s = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
    f_c = np.sqrt(ecc) * np.cos(np.deg2rad(omega))

    def _model(x, fw):
        return rv(x, 
                  radius_1=r1oa,
                  radius_2=r2oa,
                  period=period.value, 
                  t_zero=t0, 
                  incl=incl, 
                  sbratio=0,
                  a=a, 
                  q=q, 
                  f_s=f_s, 
                  f_c=f_c, 
                  ld_1=ld,
                  ldc_1=ustar,
                  flux_weighted=fw,
                  vsini_1=vsini,
                  lambda_1=ell,
                  t_exp=texp,
                  n_int=oversample,
                  grid_1='very_sparse',
                  grid_2='very_sparse'
                  )[0] * 1e3

    if slope:
        model = _model(x, True)
    else:
        model = _model(x, True) - _model(x, False)

    return model


def get_14_transit_duration(P, roa, ror, incl, ecc=0, omega=90):

    incl = np.deg2rad(incl)
    omega = np.deg2rad(omega)

    b = (np.cos(incl) / roa *
            ( 
                (1 - ecc**2) / (1 + ecc * np.sin(omega)) 
                )
            )

    return (P / np.pi * np.arcsin(
        roa * np.sqrt(
            (1 + ror)**2 - b**2
            ) / 
        np.sin(incl)
        ) *
     np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega))
    )

def get_23_transit_duration(P, roa, ror, incl, ecc=0, omega=90):

    incl = np.deg2rad(incl)
    omega = np.deg2rad(omega)

    b = (np.cos(incl) / roa *
            ( 
                (1 - ecc**2) / (1 + ecc * np.sin(omega)) 
                )
            )

    return (P / np.pi * np.arcsin(
        roa * np.sqrt(
            (1 - ror)**2 - b**2) / np.sin(incl)
        ) *
     np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega))
    )

def get_transit_mask(x, dur, texp=0, ref=0):
    return ((x - ref) > -0.5 * (dur + texp)) & ((x - ref) < 0.5 * (dur + texp))

