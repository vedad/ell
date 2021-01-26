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


def plot_fit(x, y, yerr, omc, xmod, ymod,
            z=None,
             samples=None, mod_sd=None,
                period=1,
                figsize=None, ylim=None,
                gridspec_kwargs={},
                style='default'):

    if isinstance(x, np.ndarray):
        x = [x]
    if isinstance(y, np.ndarray):
        y = [y]
    if isinstance(yerr, np.ndarray):
        yerr = [yerr]
    if isinstance(omc, np.ndarray):
        omc = [omc]
    if z is not None and isinstance(z, np.ndarray):
        z = [z]

    n = len(x)

    plt.style.use(style)


    if samples is not None:
        samples = np.atleast_2d(samples)


    if figsize is None:
        fs = figsize
    elif None in figsize:

        fig    = plt.figure()
        w, h   = figsize
        _w, _h = fig.get_size_inches()

        w = _w if w is None else w
        h = _h if h is None else h

        figsize = (w, h)

    ncols = 1
    gridspec_kw = {
                  'height_ratios':[3,1], 
                  'hspace':0.03
                  }

    if z is not None:
        ncols += 1
        gridspec_kw['width_ratios'] = [20,1]
        gridspec_kw['wspace'] = 0.02


    gridspec_kw.update(gridspec_kwargs)

    gs = GridSpec(2, ncols, **gridspec_kw)
    fig = plt.figure(figsize=figsize)
#    fig, axes = plt.subplots(2, ncols,
##                         sharex=True, 
#                         figsize=figsize,
#                         gridspec_kw=gridspec_kw
#                         )
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex=ax1)

    if z is not None:
        axc = fig.add_subplot(gs[:,1])
#    if z is None:
#        ax1 = fig.add_subplot(gs[0,0])
#        ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
##        ax1 = axes[0]
##        ax2 = axes[1]
#    else:
#        ax1 = axes[0,0]
#        ax2 = axes[1,0]
#        axc = axes[0,1]

    ax1.set_ylabel('occulted surface RV (km/s)')
    ax1.tick_params(axis='x', which='both', labelbottom=False)

    if ylim is None:
        offset = np.median(yerr)
        ylim = (ymod.min()-offset, ymod.max()+offset)
    ax1.set_ylim(ylim)


    ax2.tick_params(axis='x', which='both', top=True)
    xlabel = 'phase'
    if period is not None:
        xlabel += ' (hours)'
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(-5*np.median(yerr), 5*np.median(yerr))

    if period is not None:
        period *= 24

    if n == 1:
        markers = ['.']
    else:
        markers = ['v', '^', '.', 's', 'd']

    colors = ['C{0}'.format(i) for i in range(5)]

    if z is None:
        for i in range(n):
            kwargs = {'capsize':0, 'fmt':'none', 'color':colors[i],
                    'alpha':0.5}
            ax1.errorbar(x[i]*period, y[i], yerr=yerr[i], **kwargs)
            ax2.errorbar(x[i]*period, omc[i], yerr=yerr[i],
                    **kwargs)

            kwargs = {'fmt':markers[i], 'color':colors[i]}
            ax1.errorbar(x[i]*period, y[i], **kwargs)
            ax2.errorbar(x[i]*period, omc[i], **kwargs)

#        ax1.plot(xmod*period, ymod, color='C1')

    else:
        for i in range(n):

            # plot errorbars
            kwargs = {'capsize':0, 'fmt':'none', 'color':'#aaaaaa', 'alpha':1.0,
                    'zorder':-100}
            ax1.errorbar(x[i]*period, y[i], yerr=yerr[i],
                    **kwargs)
            ax2.errorbar(x[i]*period, omc[i], yerr=yerr[i],
                    **kwargs)

            kwargs = {'marker':markers[i],# 's':40,
                    'edgecolors':'none', 'c':z[i], 'cmap':cm.inferno,
                    'vmin':0, 'vmax':1, 'zorder':100}
            im = ax1.scatter(x[i]*period, y, 
                    **kwargs)
            imr = ax2.scatter(x[i]*period, omc[i], **kwargs)

        ax1.plot(xmod*period, ymod, color='C0')

        cbar = fig.colorbar(im, cax=axc)
        cbar.set_label('$\langle \mu \\rangle$')
        xmajorlocator = plticker.MaxNLocator(prune='both', nbins=6)
        cbar.set_ticks(xmajorlocator)
        cbar.update_ticks()
#                    marker=markers[n],
#                            s=markersize[n],
#                            edgecolors='none',
#                            c=relo.mu_avg[inds[n]], cmap=cmap, vmin=0.0, vmax=1.0,
#                            label=labels[n],
#                            zorder=100)
#            imr = axr.scatter(p[inds[n]]*p2h, resid[inds[n]], marker=markers[n],
#                              s=markersize[n],
#                              edgecolors='none',
#                              c=relo.mu_avg[inds[n]],cmap=cmap, vmin=0.0, vmax=1.0,
#                              label=labels[n],
#                              zorder=100)



    if samples is not None:
        for i in range(samples.shape[0]):
            ax1.plot(xmod*period, samples[i], color='C1', lw=0.5, alpha=0.2)

    if mod_sd is not None:
        ax1.fill_between(xmod*period, ymod-mod_sd, ymod+mod_sd, alpha=0.5,
        c='C1', lw=0, edgecolor='none')

    ax2.axhline(0, c="#aaaaaa", lw=2)

    return fig

