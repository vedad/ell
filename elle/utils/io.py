#!/usr/bin/env python

from __future__ import (print_function, division)

import numpy as np
from astropy.io import fits

__all__ = ["get_header", "read"]

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
    def __init__(self):
        raise NotImplementedError
