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



def plot_ccfs(rv, ccf, ccf_err=None, model=None, mask=None):#fwhm=None, sigma=None, k=3, model=None):

    m = mask

#    rv  = np.atleast_1d(rv)
#    ccf = np.atleast_2d(ccf)

#    if fwhm is not None and sigma is None:
#        fwhm = np.atleast_1d(fwhm)
#        sigma = fwhm / (2*np.sqrt(2 * np.log(2)))
#
#    if sigma is not None and fwhm is None:
#        sigma = np.atleast_1d(sigma)

    n     = ccf.shape[0]
    nrows = int(np.ceil(n/3))
    ncols = 3
    gs    = GridSpec(nrows, ncols, hspace=0.2, wspace=0.2)
    fig   = plt.figure(figsize=(10,int(3*nrows)))
    
    i = 0

    if ccf_err is not None:
        ymin = np.min(ccf) - np.max(ccf_err)*3
        ymax = np.max(ccf) + np.max(ccf_err)*3
    
    for row in range(nrows):
        for col in range(ncols):  
            if i == n:
                break

            ax = fig.add_subplot(gs[row, col])

            # find RV at CCF minimum
#            rv0 = rv[i][np.argmin(ccf[i])]

            # define continuum range
#            if fwhm is None and sigma is None:
#                c = 1
#            else:
#                m = (rv[i] > (rv0 - 3*sigma[i])) & (rv[i] < (rv0 + 3*sigma[i]))
#                c = np.median(ccf[i][~m])
            
#            ccf[i] /= c

            ax.plot(rv[i], ccf[i], 'C0', zorder=0)
            
            if ccf_err is not None:
                ax.errorbar(rv[i], ccf[i], yerr=ccf_err[i], capsize=0,
                        fmt='.', ms=2, color='C0', zorder=-100)

#            if sigma is not None and fwhm is not None:
            if mask is not None:
                ax.plot(rv[i][m[i]], ccf[i][m[i]], 'C1', zorder=100)

            if model is not None:
                ax.plot(rv[i], model[i], 'C1')
            
#            if model is not None:
#                ax.plot(rv[i], model[i], 'C1')
            if ccf_err is not None:
                ax.set_ylim(ymin, ymax)

            i += 1

    return fig

def plot_trace(phase, rv, ccf, transit_mask, 
               bad=None,
               duration_14=None, duration_23=None,
               period=None,
               out_offset=None,
               interpolate_trace=True,
               use_percentiles=False, cmap='RdBu_r',
               style='default', figsize=(6,6), gridspec_kwargs={},
               show_legend=False,
               trace_ylabel='phase',
               flux_ylabel='flux'):

    if bad is not None:
        keep = np.arange(len(transit_mask))
        bad  = np.atleast_1d(bad)
        keep = np.setdiff1d(keep, bad)

        phase        = phase[keep]
        ccf          = ccf[keep]
        transit_mask = transit_mask[keep]


    plt.style.use(style)

    gridspec_kw = {
                  'height_ratios':[2,1], 
                  'width_ratios':[20,1], 
                  'wspace':0.03, 
                  'hspace':0.03
                  }

    gridspec_kw.update(gridspec_kwargs)

    m       = transit_mask
    ccf_in  = ccf[m]
    ccf_out = ccf[~m]

    if figsize is None:
        fs = figsize
    elif None in figsize:

        fig    = plt.figure()
        w, h   = figsize
        _w, _h = fig.get_size_inches()

        w = _w if w is None else w
        h = _h if h is None else h

        figsize = (w, h)



    fig, axes = plt.subplots(2, 2,
#                         sharex=True, 
                         figsize=figsize,
                         gridspec_kw=gridspec_kw
                         )


    ax_res   = axes[0,0]
    ax_trace = axes[1,0]
    ax_c     = axes[1,1]
    axes[0,1].axis('off')

    ax_res.set_xlim([rv.min(),rv.max()])
    ax_res.set_ylabel(flux_ylabel)

    ax_trace.set_xlim([rv.min(),rv.max()])
    ax_trace.set_ylim([phase.min(), phase.max()])
    ax_trace.set_xlabel('radial velocity (km/s)')
    ax_trace.set_ylabel(trace_ylabel)

    # plot out-of-transit residuals in gray with arbitrary offset
    if out_offset is not None:
        in_min = out_offset
    else:
        in_min = np.abs(np.min(ccf_in))
        if any(~m):
            in_min += np.abs(np.max(ccf_out))

    # alpha gradient for in-transit CCFs to show time evolution
    colorgrad = cm.get_cmap('Greens')(np.linspace(0.4,1,len(phase[m])))
    for i in range(ccf_in.shape[0]):

        label = None
        if i < (ccf_in.shape[0] - 1):
            label = None
        else:
            label = 'in transit'

        ax_res.plot(rv, ccf_in[i], lw=0.5, c=colorgrad[i],
                    label=label, rasterized=True)

    for i in range(ccf_out.shape[0]):

        label = None
        if i < (ccf_out.shape[0] - 1):
            label = None
        else:
            label = 'out of transit'

        ax_res.plot(rv, ccf_out[i]-in_min, lw=0.5, c='gray',
                    label=label, rasterized=True)
#        ax_res.plot(rv, ccf_out[i], lw=0.5, c='gray',
#                    label=label, rasterized=True, zorder=-300, alpha=0.5)


    if show_legend:
        ax_res.legend(frameon=False)

    if use_percentiles:
        vmin = np.percentile(ccf_in, 20)
        vmax = np.percentile(ccf_in, 95)
    else:
        vmin, vmax = ccf.min(), ccf.max()

    norm = colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
    cmap = cm.get_cmap(cmap)

#    if trace == 'contour':
#        X, Y = np.meshgrid(rv, phase)
#        im = ax_trace.contourf(X, Y, ccf, 300, cmap=cmap, rasterized=True, 
#                norm=norm)
#    elif trace == 'mesh':
    if interpolate_trace:
        shading = 'gouraud'
    else:
        shading = 'flat'

    im = ax_trace.pcolormesh(rv+np.zeros_like(ccf), phase[:,None], ccf, 
            cmap=cmap, linewidth=0, rasterized=True, norm=norm,
            shading=shading)
#    im.set_edgecolor('face')
#    else:
#        raise ValueError("trace value '{0}' not recognised, "
#                         "use 'contour' or 'mesh'")
#        sys.exit()

    cbar = fig.colorbar(im, cax=ax_c)
#    cbar.set_label('flux (ppt)')

    ax_trace.axvline(0, linestyle='dotted', color='#aaaaaa', linewidth=1.0,
            rasterized=True)
    ax_trace.axhline(0, linestyle='dotted', color='#aaaaaa', linewidth=1.0,
            rasterized=True)

#    if duration_14 is not None and period is not None:
    if duration_14 is not None:
        ax_trace.axhline(-0.5*duration_14, linestyle='solid', 
                color='#aaaaaa', linewidth=1.0, rasterized=True)
        ax_trace.axhline(0.5*duration_14, linestyle='solid', 
                color='#aaaaaa', linewidth=1.0, rasterized=True)

#        ax_trace.axhline(-0.5*duration_14/period, linestyle='solid', 
#                color='#aaaaaa', linewidth=1.5)
#        ax_trace.axhline(0.5*duration_14/period, linestyle='solid', 
#                color='#aaaaaa', linewidth=1.5)

#    if duration_23 is not None and period is not None:
    if duration_23 is not None:
        if np.isfinite(duration_23):
            ax_trace.axhline(-0.5*duration_23, linestyle='dashed', 
                    color='#aaaaaa', linewidth=1.0, rasterized=True)
            ax_trace.axhline(0.5*duration_23, linestyle='dashed',
                    color='#aaaaaa', linewidth=1.0, rasterized=True)

#            ax_trace.axhline(-0.5*duration_23/period, linestyle='dashed', 
#                    color='#aaaaaa', linewidth=1.5)
#            ax_trace.axhline(0.5*duration_23/period, linestyle='dashed',
#                    color='#aaaaaa', linewidth=1.5)

    return fig


#colors = ["#A01810", "#E4704A", "#1E4864", "#2F90A7", "#BD4E31"]
def plot_surface_velocity(x, y, yerr,
            xmod=None,
            ymod=None,
            ymod_sd=None,
            ymod_kmax=3,
            samples=None,
            ymod_color="#A01810",
            samples_cmap="Reds",
            residual=None,
            style='paper',
            figsize=(5, 3.8)
            ):

    if ymod is not None and samples is not None:
        raise ValueError("only `ymod` or `samples` may be used, not both")

    plot_model = (xmod is not None) & (ymod is not None)
    plot_samples = samples is not None
    plot_residual = residual is not None

    if isinstance(x, np.ndarray):
        x = [x]
    if isinstance(y, np.ndarray):
        y = [y]
    if isinstance(yerr, np.ndarray):
        yerr = [yerr]
    if plot_residual and isinstance(residual, np.ndarray):
        residual = [residual]

    n = len(x)

    markers = ['.', 'd', 'x', '^', 's']

#    colors = ["#A01810", "#E4704A", "#1E4864", "#2F90A7", "#BD4E31"] # org blue/orange
    # blue/orange in new order
    colors = ["#1E4864",  "#E4704A",  "#2F90A7",  "#BD4E31"]
#    colors = ["#2F90A7", "#E4704A", "#1E4864",   "#BD4E31"]

    sizes = [6, 4, 5, 5, 5]

    if plot_residual:
        rows = 2
        gridspec_kw = {
              'height_ratios':[3,1], 
              'hspace':0.03,
              'wspace':0.02
              }
    else:
        rows = 1
        gridspec_kw = None

    if figsize is None:
        fs = figsize
    elif None in figsize:

        fig    = plt.figure()
        w, h   = figsize
        _w, _h = fig.get_size_inches()

        w = _w if w is None else w
        h = _h if h is None else h

        figsize = (w, h)

    fig, ax = plt.subplots(rows, 1, gridspec_kw=gridspec_kw, squeeze=False,
            figsize=figsize, sharex=True)

    ax[0][0].set_ylabel('surface radial velocity (km/s)')
    ax[0][0].set_xlim(xmod.min(), xmod.max())

    if plot_residual:
        ax[0][0].tick_params(axis='x', which='both', labelbottom=False)
        ax[1][0].set_xlabel('phase')
        ax[1][0].set_ylabel("O - C (km/s)")
        ax[1][0].axhline(0, c="#aaaaaa", lw=1.5)
        for i in range(n):
            ax[1][0].errorbar(
                    x[i],
                    residual[i],
                    yerr=yerr[i],
                    capsize=0,
                    markersize=sizes[i],
                    c=colors[i],
                    fmt=markers[i],
                    elinewidth=0.5)
    else:
        ax[0][0].set_xlabel('phase')

    for i in range(n):
        ax[0][0].errorbar(
                x[i], 
                y[i], 
                yerr=yerr[i], 
                capsize=0, 
                markersize=sizes[i],
                c=colors[i],
                fmt=markers[i],
                elinewidth=0.5)

    if plot_model and not plot_samples:
        ax[0][0].plot(xmod, ymod, color=ymod_color, lw=1.5)

        if ymod_sd is not None:

            for k in range(1,ymod_kmax+1):
                ax[0][0].fill_between(
                        xmod,
                        ymod - k * ymod_sd,
                        ymod + k * ymod_sd,
                        color=ymod_color,
                        lw=0,
                        alpha=0.4 - k * 0.1)

                ax[1][0].fill_between(
                        xmod,
                        - k * ymod_sd,
                        k * ymod_sd,
                        color=ymod_color,
                        lw=0,
                        alpha=0.4 - k * 0.1)

    elif plot_samples:
        cmap = plt.get_cmap(samples_cmap)

        percs = np.linspace(51, 99, 100)
        _colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
        for i, p in enumerate(percs[::-1]):
            upper = np.percentile(samples, p, axis=0)
            lower = np.percentile(samples, 100-p, axis=0)
            color_val = _colors[i]
            ax[0][0].fill_between(
                    xmod,
                    upper, 
                    lower,
                    color=cmap(color_val), 
                    alpha=0.8, 
                    zorder=-200)


    return fig

