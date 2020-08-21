import numpy as np
import struct
import collections
import pickle
import re
import requests
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullFormatter
from scipy.signal import resample, resample_poly
from pathlib import Path
import pretty_errors
from obspy.signal.tf_misfit import plot_tfr, em, pm, tem, tpm, fem, fpm, tfem, tfpm
from post_processing.la_habra import *
from filter_BU import filt_B
import my_pyrotd

def resize(vel, vel_rec, dt=None):
    """Reshape the vel and vel_rec accoring with dt
    Input
    -----
        vel : dict with keys ['dt', 't', 'X', 'Y', 'Z']
        vel_rec : dict with keys ['dt', 't', 'X', 'Y', 'Z']
            The same as vel, but for recordings
    """
    length = int(len(vel['t']) * vel['dt'] // dt)
    length_rec = int(len(vel_rec['t']) * vel_rec['dt'] // dt)
    print(f'Length of time steps = {length}')
    if length > len(vel['t']):
        print(f"Upsample may introduce alias. Use larger dt_sample instead.\n")
        return
    for comp in 'XYZ':
        # First trim vel and vel_rec so that their lengths are the same
        # Then resample to dt
        if length > length_rec:
            vel[comp] = vel[comp][vel['t'] <= vel_rec['t'][-1]]
            vel[comp] = resample(vel[comp], length_rec)
            vel_rec[comp] = resample(vel_rec[comp], length_rec)
        else:
            vel_rec[comp] = vel_rec[comp][vel_rec['t'] <= vel['t'][-1]]
            vel_rec[comp] = resample(vel_rec[comp], length)
            vel[comp] = resample(vel[comp], length)


def plot_obspy_tf_misfit(misfit, comps='XYZ', left=0.1, bottom=0.1,
                    h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6, w_cb=0.01,
                    d_cb=0.0, show=True, plot_args=['k', 'r', 'b'], ylim=0.,
                    clim=0., cmap='RdBu_r'):
    """Funciton to plot tf_misfit, revised from Obspy
    """
    from matplotlib.ticker import NullFormatter
    figs = []
    t = misfit['t']
    f = misfit['f']
    ntr = len(comps)
    for itr, comp in enumerate(comps):
        fig = plt.figure(dpi=400)
        data = misfit[comp]

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3, w_2, h_1])
        ax_sig.plot(t, data['syn'], plot_args[0], label='syn')
        ax_sig.plot(t, data['rec'], plot_args[1], label='data')
        ax_sig.legend(loc=1, ncol=2)

        # plot TEM
        if 'tem' in data:
            ax_tem = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3, w_2, h_2])
            ax_tem.plot(t, data['tem'], plot_args[2])

        # plot TFEM
        if 'tfem' in data:
            ax_tfem = fig.add_axes([left + w_1, bottom + h_1 + 2 * h_2 + h_3, w_2,
                                    h_3])

            img_tfem = ax_tfem.pcolormesh(t, f, data['tfem'], cmap=cmap)
            img_tfem.set_rasterized(True)
            ax_tfem.set_yscale("log")
            ax_tfem.set_ylim(fmin, fmax)

        # plot FEM
        if 'fem' in data:
            ax_fem = fig.add_axes([left, bottom + h_1 + 2 * h_2 + h_3, w_1, h_3])
            ax_fem.semilogy(data['fem'], f, plot_args[2])
            ax_fem.set_ylim(fmin, fmax)

        # plot TPM
        if 'tpm' in data:
            ax_tpm = fig.add_axes([left + w_1, bottom, w_2, h_2])
            ax_tpm.plot(t, data['tpm'], plot_args[2])

        # plot TFPM
        if 'tfpm' in data:
            ax_tfpm = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])

            img_tfpm = ax_tfpm.pcolormesh(t, f, data['tfpm'], cmap=cmap)
            img_tfpm.set_rasterized(True)
            ax_tfpm.set_yscale("log")
            ax_tfpm.set_ylim(f[0], f[-1])

        # add colorbars
        ax_cb_tfpm = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                                   w_cb, h_2 + h_3])
        fig.colorbar(img_tfpm, cax=ax_cb_tfpm)

        # plot FPM
        if 'fpm' in data:
            ax_fpm = fig.add_axes([left, bottom + h_2, w_1, h_3])
            ax_fpm.semilogy(data['fpm'], f, plot_args[2])
            ax_fpm.set_ylim(fmin, fmax)

        # set limits
        ylim_sig = np.max([np.abs(data['rec']).max(), np.abs(data['rec']).max()]) * 2.5
        ax_sig.set_ylim(-ylim_sig, ylim_sig)

        if ylim == 0.:
            ylim = np.max([np.abs(data[key]).max() for key in ['tem', 'tpm', 'fem', 'fpm'] \
                           if key in data]) * 1.1

        ax_tem.set_ylim(-ylim, ylim)
        ax_fem.set_xlim(-ylim, ylim)
        ax_tpm.set_ylim(-ylim, ylim)
        ax_fpm.set_xlim(-ylim, ylim)

        ax_sig.set_xlim(t[0], t[-1])
        ax_tem.set_xlim(t[0], t[-1])
        ax_tpm.set_xlim(t[0], t[-1])

        if clim == 0.:
            clim = np.max([np.abs(data['tfem']).max(), np.abs(data['tfpm']).max()])

        img_tfpm.set_clim(-clim, clim)
        img_tfem.set_clim(-clim, clim)

        # add text box for EM + PM
        textstr = f"{comp}-component\nEM = {data['em']: .2f}\nPM = {data['pm']: .2f}"
        props = dict(boxstyle='round', facecolor='white')
        ax_sig.text(-0.3, 0.5, textstr, transform=ax_sig.transAxes,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=props)

        ax_tpm.set_xlabel('time (s)')
        ax_fem.set_ylabel('frequency (Hz)')
        ax_fpm.set_ylabel('frequency (Hz)')

        # add text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_tfem.text(0.95, 0.85, 'TFEM', transform=ax_tfem.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tfpm.text(0.95, 0.85, 'TFPM', transform=ax_tfpm.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tem.text(0.95, 0.75, 'TEM', transform=ax_tem.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_tpm.text(0.95, 0.75, 'TPM', transform=ax_tpm.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fem.text(0.9, 0.85, 'FEM', transform=ax_fem.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fpm.text(0.9, 0.85, 'FPM', transform=ax_fpm.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)

        # remove axis labels
        ax_tfpm.xaxis.set_major_formatter(NullFormatter())
        ax_tfem.xaxis.set_major_formatter(NullFormatter())
        ax_tem.xaxis.set_major_formatter(NullFormatter())
        ax_sig.xaxis.set_major_formatter(NullFormatter())
        ax_tfpm.yaxis.set_major_formatter(NullFormatter())
        ax_tfem.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs


def comp_obspy_tf_misfit(model, site_name, dt, comps='XYZ',
                         fmin=0.15, fmax=5, nf=128, vel=None, vel_rec=None, plot=False):
    """See doc in `comp_obspy_tf_misfits` for details
    """
    if vel is None:
        with open('results/vel_syn.pickle', 'rb') as fid:
            vel_syn = pickle.load(fid)
        vel = vel_syn[model][site_name]
        vel_rec = vel_syn["rec"][site_name]

    resize(vel, vel_rec, dt)
    res = {}
    res['dt'] = dt
    res['f'] = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    for comp in comps:
        res[comp] = {}
        res[comp]['syn'] = vel[comp]
        res[comp]['rec'] = vel_rec[comp]
        res[comp]['em'] = em(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['pm'] = pm(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['tfem'] = tfem(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['tfpm'] = tfpm(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['tem'] = tem(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['tpm'] = tpm(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['fem'] = fem(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
        res[comp]['fpm'] = fpm(vel[comp], vel_rec[comp], dt, fmin, fmax, nf)
    res['t'] = np.arange(res[comp]['tem'].shape[-1]) * dt
    if plot:
        plot_obspy_tf_misfit(res, comps=comps)
    return res



def comp_obspy_tf_misfits(models, dt, comps='XYZ', fmin=0.15, fmax=5, nf=128):
    """Compute obspy tf_misfits for models
    Input
    -----
        models : list of string
            Serves as keys in vel
        dt : float
            Uniform time spacing
        comps : ['XYZ']
            Components to compute
        fmin, fmax : float, float
            Frequency range
        nf : int
            Number of frequencies

    Output
    ------
        misfit : nested dict
            1st-layer keys: models
            2nd-layer keys: site_name
            3rd-layer keys: ['dt', 't', 'f', 'X', 'Y', 'Z']
    
    Note
    ----
        Key input is vel_syn, which records the synthetics and recordings.
        vel_syn : nested dict
            1st-layer keys: model
            2st-layer keys: site_name
            3rd-layer keys: ['dt', 't', 'X', 'Y', 'Z']
    """
            
    misfit = {}
    with open('results/vel_syn.pickle', 'rb') as fid:
        vel_syn = pickle.load(fid)

    for model in models:
        misfit[model] = {}
        vel = vel_syn[model]
        vel_rec = vel_syn['rec']
        for site_name in vel_syn[model].keys():
            print(f'{model}: {site_name}')
            misfit[model][site_name] = comp_obspy_tf_misfit( \
                    model, site_name, dt, comps=comps, fmin=fmin, fmax=fmax,
                    nf=nf, vel=vel[site_name], vel_rec=vel_rec[site_name])
    return misfit
