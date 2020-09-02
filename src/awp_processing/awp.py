'''The main analyzing subroutine
TODO:
    1. Use multiprocessing to accelerate?
    2. Add plot subroutines, maybe in a seperate .py
'''
from pathlib import Path
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, ScalarFormatter, NullFormatter
from collections import defaultdict
from obspy.signal.detrend import polynomial
from scipy.signal import detrend
from obspy.signal.konnoohmachismoothing import calculate_smoothing_matrix
import sys

import my_pyrotd

from .utils import *
from .read_params import read_params
from .filter_BU import filt_B


class Scenario:
    def __init__(self, model="", case="", conf_file='param.sh'):
        '''
        Input:
            case (str): different cases of a model, if exists
        '''
        self.model = model
        self.case = "output_sfc" if not case else f"{case}"
        self.output_dir = Path(self.model, self.case)
        self.cfg = AttrDict(read_params(Path(model, conf_file)))
        self.shape_of_block()
        
 
    def shape_of_block(self):
        '''Shape of block, (nx, ny, nz)'''
        self.cfg.nx = [0] * self.cfg.g
        self.cfg.ny = [0] * self.cfg.g
        self.cfg.nz = [0] * self.cfg.g
        print(f'Number of blocks: {self.cfg.g}')
        for g in range(self.cfg.g):
            # print(type(self.cfg.nedx), type(self.cfg.nskpx), self.cfg.nskpx)
            self.cfg.nx[g] = (self.cfg.nedx[g] - self.cfg.nbgx[g]) // self.cfg.nskpx[g] + 1
            self.cfg.ny[g] = (self.cfg.nedy[g] - self.cfg.nbgy[g]) // self.cfg.nskpy[g] + 1
            self.cfg.nz[g] = (self.cfg.nedz[g] - self.cfg.nbgz[g]) // self.cfg.nskpz[g] + 1
        
    
    def reindex(self, i, comp='x', block=0):
        '''Reindex i along comp direction in block
        It's possible to specify the volumn size & location, which is different from the whole domain. This function computes the corresponding size of output volumn and index of input i.
        Return
        ------
            ret[0] : corresponding index
            ret[1] : corresponding size
        '''
        bg = self.cfg['nbg' + comp][block]
        ed = self.cfg['ned' + comp][block]
        skip = self.cfg['nskp' + comp][block]
        if i < 0:
            print("Likely illegal input index")
            return i, (ed - bg) // skip + 1
        if i < bg or i > ed:
            print(f"i{comp} = {i}: outside output range. Rounded to {min(max(i, bg), ed)}")
            i = min(max(i, bg), ed)
        shift = i - bg
        if shift % skip != 0:
            print(f"i{comp} = {i}: rounded using nearest grid {int(round(shift / skip))}")
        return int(round(shift / skip)), (ed - bg) // skip + 1


    def reindex_block(self, ix, iy, iz=1, block=0):
        '''Shift and compress index, return 0-indexed'''
        ix, nx = self.reindex(ix, 'x')
        iy, ny = self.reindex(iy, 'y')
        iz, nz = self.reindex(iz, 'z')
        return ix, nx, iy, ny, iz, nz

    
    def read_slice(self, t, ix=-1, iy=-1, iz=-1, block=0, comp="X"):
        '''Read wave field snapshot
        Input:
            t: time to read
            ix, iy, iz (int): Indices of cross-section in the original mesh, 1-indexed
            block: The index of the block, 0-indexed
            comp: different cases of a model, if exists
        '''
        
        print(f"Params: tmax={self.cfg.tmax}, dt={self.cfg.dt}, wstep={self.cfg.wstep}, tskip={self.cfg.tskip}")
        if t < 0 or t >= self.cfg.tmax:
            print(f"Range of t: [0, {self.cfg.tmax}]")
            sys.exit(-1)
        ix, nx, iy, ny, iz, nz = self.reindex_block(ix, iy, iz, block=block)

        resi, it = np.divmod(int(t / self.cfg.dt / self.cfg.tskip), self.cfg.wstep) 
        fnum = (resi + 1) * self.cfg.wstep * self.cfg.tskip 
        file_name = f'{self.output_dir}/S{comp}_{block}_{fnum:07d}'
        print(f'\rit = {it}, t = {t}s / {self.cfg.tmax}s, {it * self.cfg.tskip} / {self.cfg.wstep} in file = {file_name}', end="\r", flush=True)
        
        v = np.fromfile(file_name, dtype='float32', count=nz * ny * nx,
                        offset=it * nx * ny * nz * 4).reshape(nz, ny, nx)
        idx = np.where(np.isnan(v))
        if np.isnan(v).any():
            print(f"\n{len(idx[0])} NANs founded\n")
        
        print(f"\nShape of velocity output: (nz, ny, nx) = ({nz}, {ny}, {nx})")
        if ix >= 0:
            v = v[:, :, ix]
            print(f'\nThe x_index is: {ix * self.cfg.nskpx[block]} / {self.cfg.nedx[block]} --> ({v.shape})\nOriginal vmax = {np.max(v):.5e}')
        elif iy >= 0:
            v = v[:, iy, :]
            print(f'\nThe y_index is: {iy * self.cfg.nskpy[block]} / {self.cfg.nedy[block]} --> ({v.shape})\nOriginal vmax = {np.max(v):.5e}')
        elif iz >= 0:
            v = v[iz, :, :]
            print(f'\nThe z_index is: {iz * self.cfg.nskpz[block]} / {self.cfg.nedz[block]} --> ({v.shape})\nOriginal vmax = {np.max(v):.5e}')
        return v.copy()
    
    
    def read_syn(self, ix, iy, iz=1, block=0, comps='xyz'):
        '''Read synthetics
        Input
        -----
            ix, iy, iz : int
                Indices of site in the original mesh, 1-indexed
            block : int
                The index of partition from surface to bottom
        Output
        ------
            v : record array
                Three-component velocities and dt, and 'psa' if comp_psa
        '''
        
        ix, nx, iy, ny, iz, nz = self.reindex_block(ix, iy, iz, block=block)
        v = defaultdict(list)
        print(f"ix={ix}, iy={iy}, iz={iz}")
        skips = [4 * (j * nz * ny * nx + 
                      iz * ny * nx + 
                      iy * nx + ix) for j in range(self.cfg.wstep)]
        dt = self.cfg.dt * self.cfg.tskip 
        nfile = int(self.cfg.tmax / dt) // self.cfg.wstep
        file_step = self.cfg.wstep * self.cfg.tskip
        for comp in comps:
            comp = comp.lower()
            v[comp] = np.zeros((int(self.cfg.tmax / dt),), dtype='float32')
            for i in range(1, nfile + 1):
                with open(f'{self.output_dir}/S{comp.upper()}_{block}_{i * file_step:07d}', 'rb') as fid:
                    chunk = (i - 1) * self.cfg.wstep
                    for j in range(self.cfg.wstep):
                        fid.seek(skips[j], 0)
                        v[comp][chunk + j] = np.frombuffer(fid.read(4), dtype='float32')
                        if np.isnan(v[comp]).any():
                            print(f"\nNAN in file {self.output_dir}/S{comp.upper()}_{block}_{i * file_step}\n")
                            return None
            v[comp] = np.array(v[comp])

        v['dt'] = dt
        v['t'] = np.arange(len(v[comps[0]])) * dt

        return AttrDict(v)


    @staticmethod
    def taper(data, dt, taper_seconds=0, pad_seconds=5):
        """taper data at the end, then padding
        Input
        -----
            data : list or 2Darray with last dimension of time series
            dt : float
            taper_seconds : float
                the length of time series to taper_seconds
            pad_seconds : float
                the length of padding in the end
        Output
        ------
            data : the same shape as original data
        """

        npts = int(2 * taper_seconds / dt +1)
        npad = int(pad_seconds / dt)
        window = np.hanning(npts)[-int(taper_seconds / dt + 1):]
        if len(data.shape) > 1:
            for i in range(data.shape[1]):      
                data[i, -len(window):] *= window
            data = np.pad(data, ((0, 0), (0, npad)))  # Pad zeros
        else:
            data[-len(window):] *= window  # taper
            data = np.pad(data, (0, npad))
        return data


    @staticmethod
    def preproc_vel(v, comps='xyz', lowcut=0.5, highcut=12, filt=False, detrd=True, taper=0, baseline=False):
        '''Pre-processing velocities
        Input:
            v (dict): velocities, 3-components and dt
            highcut (float): highcut frequency for filtering
        '''
        for c in comps:
            if detrd:
                v[c] = detrend(v[c].astype(np.float64), overwrite_data=True)
            if filt:
                v[c] = filt_B(v[c], 1 / v['dt'], lowcut=lowcut, highcut=highcut, order=5)
            if taper > 0:
                v[c] = taper(v[c], v['dt'], taper_seconds=taper)
            disp = np.cumsum(v[c]) * v['dt']
            if baseline:
                disp = polynomial(disp, order=2)
            v[c] = np.gradient(disp, v['dt'])
        if taper > 0:
            v['t'] = np.arange(len(v[comps[-1]])) * v['dt']
        return v


    def comp_spec(self, ix, iy, iz=1, fmax=0, block=0, lowcut=0.5, highcut=12, comps='xy', metric='vel', filt=False, detrd=True, taper=0, baseline=True, smoother=None, comp_psa=False, osc_freqs=np.logspace(-1, 1.3, 181), osc_damping=0.05, return_vel=False):
        '''Compute spectrum of velocity / acceleration, RotD50
        Input:
            ix, iy, iz : int
                Indices of site in the original mesh, 1-indexed
            fmax : float
                Maximum frequency to store outputs
            block : int
                Index of block in DM code
            highcut : float
                Highcut frequency for filtering
            comp_psa : bool 
                To compute RotD50 or not 
            osc_freqs : list
                Frequencies for RotD50, default from 0.1 to 20
            osc_damping : float
                Damping

        Output:
            fv (dict): three components spectrum and frequency
        '''
        v = self.read_syn(ix, iy, iz, block)
        v = self.preproc_vel(v, lowcut=lowcut, highcut=highcut, filt=filt, detrd=detrd, taper=taper, baseline=baseline)
        fmax = fmax or highcut  # if not define fmax, choose highcut
        fv = AttrDict()
        for c in comps:
            if metric == 'acc':
                # v[c] = np.diff(v[c], prepend=0) / v['dt']  
                v[c] = np.gradient(v[c], v['dt'])
            fv[c], fv['f'] = comp_fft(v[c], v['dt'], fmax=fmax)
        if comp_psa:
            # RotD50 
            accx = np.gradient(v['x'], v['dt'])
            accy = np.gradient(v['y'], v['dt'])
            # with keys 'osc_freq', 'percentile', 'spec_accel'
            fv['psa'] = my_pyrotd.my_calc_rotated_spec_accels(
                            v['dt'], accx, accy, osc_freqs,
                            osc_damping, percentiles=[50])

        self.smoother = calculate_smoothing_matrix(fv['f'], normalize=True)
        if return_vel:
            return v, fv
        return fv


    def help_plot_slice(self, fig, ax, im, aspect, title, xlabel, ylabel, cbar_label, orientation, fig_name, tick_limit=1e-3):
        """Provide a reusable function for three componenet slice plot
        Input
        -----
            Most variables are self-explainable.
            tick_limit : float
                Use scientific notation is smaller than threshold
        """
        ax.set_aspect(aspect)
        ax.invert_yaxis()
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.tick_params(axis='x', which='both', top=False)  # hide axis ticks
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.04, pos.y0, 0.04, pos.y1 - pos.y0])
        cbar = plt.colorbar(im, cax=cax, orientation=orientation, extend='both')
        cbar.set_label(cbar_label)
        if np.max(cbar.get_ticks()) < tick_limit:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_major_formatter(formatter)
        #    cbar.ax.set_yticklabels([f"{i:.2e}" for i in cbar.get_ticks()])
        if fig_name is not None:
            fig.savefig(f"{fig_name}", dpi=600, bbox_inches='tight', pad_inches=0.05)


    def plot_slice_x(self, t, ix=0, sy=0, sz=0, maxz=1, vmin=None, vmax=None, block=0, NPY=0, comp='Z', norm='log', space_scale='km', unit='cm/s', aspect='auto', orientation='vertical', xlabel=None, ylabel=None, mesh_file=None, fig_name=None):
        """Plot snapshot at certain x
        Input
        -----
        t : float
            time of snapshot
        ix : int
            location of snapshot
        sy, sz : int
            Index of source (y, z)
        maxz : float
            Maximu depth to plot, so as to focus on near surface structure
        vmin, vmax : float
            range of velocity to plot
        block : int
            the deepest block to plot
        NPY : int
            Partition of GPUs along y
        comp : ['X', 'Y', 'Z']
            Component of velocity
        norm : 'log' or else
            If 'log', apply log scale to image
        space_scale : ['m', 'km']
            space_scale of axis, to avoid too many digits
        unit : ['m/s', 'cm/s']
            Unit of metric to plot
        aspect : ['auto', 'equal', int]
            Aspect to transform of figure width to height
        orientation : ['vertical', 'horizontal']
            Orientation of colorbar
        xlabel, ylabel : string
            Labels of x-axis and y-axis of the image
        mesh_file : string
            Mesh file name for extracting velocity interface
        fig_name : None or string 
            If string, then save image to file
        """

        cfg = self.cfg
        dh = [cfg.dh[i] for i in range(block + 1)]
        if space_scale == 'km':
            dh = [h / 1000 for h in dh]
        sy, sz = sy * dh[0], sz * dh[0]
        mesh_bot = 0  # the bottom the each mesh
        cmap = plt.cm.get_cmap('Reds')
        cmap = plt.cm.get_cmap('RdBu_r')
        cmap.set_under('k')
        cmap.set_over('k')
        
        fig, ax = plt.subplots(dpi=400)
        fig.tight_layout()
        for i in range(block + 1):
            ix_in_block = (ix - 1) // (cfg.ratio ** i) + 1
            v = self.read_slice(t, ix=ix_in_block, block=block, comp=comp)
            if unit == 'cm/s':
                v = v * 100
            print(f"Unit converted to {unit}: vmin = {np.min(v):.3e}; vmax = {np.max(v):.3e}")
            # v[v < 0] = 0
            vmax = (1 - 0.2 * np.sign(np.max(v))) * np.max(v) if i == 0 and vmax is None else vmax
            vmin = (1 + 0.2 * np.sign(np.min(v))) * np.min(v) + vmax / 100 if i == 0 and vmin is None else vmin
            if norm == 'log':
                norm = colors.SymLogNorm(linthresh=vmax/10, linscale=1, base=10)
                #norm = colors.LogNorm(vmin=vmin)  # focus on larger values
                #norm = colors.PowerNorm(gamma=0.5)  # focus on small values
            else:
                norm = None
                
            ry = np.arange(cfg.nbgy[block] - 1, cfg.nedy[block], cfg.nskpy[block])
            rz = np.arange(cfg.nbgz[block] - 1, cfg.nedz[block], cfg.nskpz[block])
            stepy = decimate(ry)
            stepz = decimate(rz)
            idx_z = np.nonzero(rz * dh[block] + mesh_bot <= maxz)[0][::stepz]
            im = ax.pcolormesh(ry[::stepy] * dh[block], rz[idx_z] * dh[block] + mesh_bot, v[idx_z, ::stepy], vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
            if mesh_file is not None:
                mesh = np.fromfile(f'{mesh_file}_{i}', dtype='float32').reshape(cfg.z[i], cfg.y[i], cfg.x[i], cfg.nvar)[np.ix_(rz, ry, [ix - 1], [1])].squeeze()
                ax.contour(ry[::stepy] * dh[block], rz[idx_z] * dh[block] + mesh_bot, mesh[idx_z, ::stepy], 3, cmap='gray', linewidths=0.8)
            mesh_bot += (cfg.z[block] - 7) * dh[block]
            print(f'Mesh bottom of block {block} is = {mesh_bot}')
        if sy > 0 or sz > 0:
            ax.scatter(sy, sz, 150, color='g', marker='*')
            ax.axhline(sz, color='g')
        for i in range(1, NPY):
            ax.axvline(cfg.y[0] // i * dh[0], linestyle=':', linewidth=1.2, color='c')


        title = f'T = {t:.3f}s'
        xlabel = xlabel or f'X ({space_scale})'
        ylabel = ylabel or f'Y ({space_scale})'
        cbar_label = f'V{comp} ({unit})'
        self.help_plot_slice(fig, ax, im, aspect, title, xlabel, ylabel, cbar_label, orientation, fig_name, tick_limit=1e-3)
        return save_image(fig)


    def plot_slice_y(self, t, iy=0, sx=0, sz=0, maxz=1, vmin=None, vmax=None, block=0, NPX=0, comp='Z', norm='log', space_scale='km', unit='cm/s', aspect='auto', orientation='vertical', xlabel=None, ylabel=None, mesh_file=None, fig_name=None):
        """Plot snapshot at certain x
        Input
        -----
        t : float
            time of snapshot
        iy : int
            location of snapshot
        sx, sz : int
            Index of source (x, z)
        maxz : float
            Maximu depth to plot, so as to focus on near surface structure
        vmin, vmax : float
            range of velocity to plot
        block : int
            the deepest block to plot
        NPX : int
            Partition of GPUs along x
        comp : ['X', 'Y', 'Z']
            Component of velocity
        norm : 'log' or else
            If 'log', apply log scale to image
        space_scale : ['m', 'km']
            space_scale of axis, to avoid too many digits
        unit : ['m/s', 'cm/s']
            Unit of metric to plot
        aspect : ['auto', 'equal', int]
            Aspect to transform of figure width to height
        orientation : ['vertical', 'horizontal']
            Orientation of colorbar
        xlabel, ylabel : string
            Labels of x-axis and y-axis of the image
        mesh_file : string
            Mesh file name for extracting velocity interface
        fig_name : None or string 
            If string, then save image to file
        """

        cfg = self.cfg
        dh = [cfg.dh[i] for i in range(block + 1)]
        if space_scale == 'km':
            dh = [h / 1000 for h in dh]
        sx, sz = sx * dh[0], sz * dh[0]
        mesh_bot = 0  # the bottom the each mesh
        cmap = plt.cm.get_cmap('Reds')
        cmap = plt.cm.get_cmap('RdBu_r')
        cmap.set_under('k')
        cmap.set_over('k')
        
        fig, ax = plt.subplots(dpi=400)
        fig.tight_layout()
        for i in range(block + 1):
            iy_in_block = (iy - 1) // (3 ** i) + 1
            v = self.read_slice(t, iy=iy_in_block, block=block, comp=comp)
            if unit == 'cm/s':
                v = v * 100
            print(f"Unit converted to {unit}: vmin = {np.min(v):.3e}; vmax = {np.max(v):.3e}")
            # v[v < 0] = 0
            vmax = (1 - 0.2 * np.sign(np.max(v))) * np.max(v) if i == 0 and vmax is None else vmax
            vmin = (1 + 0.2 * np.sign(np.min(v))) * np.min(v) + vmax / 100 if i == 0 and vmin is None else vmin
            if norm == 'log':
                norm = colors.SymLogNorm(linthresh=vmax/10, linscale=1, base=10)
                #norm = colors.LogNorm(vmin=vmin)  # focus on larger values
                #norm = colors.PowerNorm(gamma=0.5)  # focus on small values
            else:
                norm = None
                
            rx = np.arange(cfg.nbgx[block] - 1, cfg.nedx[block], cfg.nskpx[block])
            rz = np.arange(cfg.nbgz[block] - 1, cfg.nedz[block], cfg.nskpz[block])
            stepx = decimate(rx)
            stepz = decimate(rz)
            idx_z = np.nonzero(rz * dh[block] + mesh_bot <= maxz)[0][::stepz]
            im = ax.pcolormesh(rx[::stepx] * dh[block], rz[idx_z] * dh[block] + mesh_bot, v[idx_z, ::stepx], vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
            if mesh_file is not None:
                mesh = np.fromfile(f'{mesh_file}_{i}', dtype='float32').reshape(cfg.z[i], cfg.y[i], cfg.x[i], cfg.nvar)[np.ix_(rz, [iy - 1], rx, [1])].squeeze()
                ax.contour(rx[::stepx] * dh[block], rz[idx_z] * dh[block] + mesh_bot, mesh[idx_z, ::stepx], 3, cmap='gray')
            mesh_bot += (cfg.z[block] - 7) * dh[block]
            print(f'Mesh bottom of block {block} is = {mesh_bot}')
        if sx > 0 or sz > 0:
            ax.scatter(sx, sz, 150, color='g', marker='*')
            ax.axhline(sz, color='g')
        for i in range(1, NPX):
            ax.axvline(cfg.x[0] // i * dh[0], linestyle=':', linewidth=1.2, color='c')
        
        title = f'T = {t:.3f}s'
        xlabel = xlabel or f'X ({space_scale})'
        ylabel = ylabel or f'Y ({space_scale})'
        cbar_label = f'V{comp} ({unit})'
        self.help_plot_slice(fig, ax, im, aspect, title, xlabel, ylabel, cbar_label, orientation, fig_name, tick_limit=1e-3)
        
        return save_image(fig)


    def plot_slice_z(self, t, iz=0, sx=0, sy=0, vmin=None, vmax=None, block=0, NPX=0, NPY=0, comp='X', topography=None, space_scale='km', unit='cm/s', norm='log', aspect='auto', orientation='vertical', xlabel=None, ylabel=None, fig_name=None, lahabra=False):
        """Plot snapshot at certain x
        Input
        -----
        t : float
            time of snapshot
        iz : int
            location of snapshot
        sx, sy : int
            Index of source (x, y)
        vmin, vmax : float
            range of velocity to plot
        block : int
            the deepest block to plot
        NPX, NPY : int
            Partition of GPUs along x and y
        comp : ['X', 'Y', 'Z']
            Component of velocity
        norm : 'log' or else
            If 'log', apply log scale to image
        space_scale : ['m', 'km']
            space_scale of axis, to avoid too many digits
        unit : ['m/s', 'cm/s']
            Unit of metric to plot
        aspect : ['auto', 'equal', int]
            Aspect to transform of figure width to height
        orientation : ['vertical', 'horizontal']
            Orientation of colorbar
        xlabel, ylabel : string
            Labels of x-axis and y-axis of the image
        fig_name : None or string 
            If string, then save image to file
        """

        cfg = self.cfg
        dh = [cfg.dh[i] for i in range(block + 1)]
        if space_scale == 'km':
            dh = [h / 1000 for h in dh]
        sx, sy = sx * dh[0], sy * dh[0]
        
        v = self.read_slice(t, iz=iz, block=block, comp=comp)
        if unit == 'cm/s':
            v = v * 100
        print(f"Unit converted to {unit}: vmin = {np.min(v):.3e}; vmax = {np.max(v):.3e}")
        v[v < 0] = 0
        vmax = (1 - 0.2 * np.sign(np.max(v))) * np.max(v) if vmax is None else vmax
        vmin = (1 + 0.2 * np.sign(np.min(v))) * np.min(v) + vmax / 100 if vmin is None else vmin
        if norm == 'log':
            #norm = colors.SymLogNorm(linthresh=1e-4, linscale=1e-4, base=10)
            norm = colors.LogNorm(vmin=vmin)  # focus on larger values
            #norm = colors.PowerNorm(gamma=0.5)  # focus on small values
        else:
            norm = None
            
        rx = np.arange(cfg.nbgx[block] - 1, cfg.nedx[block], cfg.nskpx[block]) * dh[block]
        ry = np.arange(cfg.nbgy[block] - 1, cfg.nedy[block], cfg.nskpy[block]) * dh[block]
        stepx = decimate(rx)
        stepy = decimate(ry)
        cmap = plt.cm.get_cmap('Reds')
        cmap.set_bad('w')
        cmap.set_under('w')
        
        fig, ax = plt.subplots(dpi=400)
        fig.tight_layout()
        
        im = ax.pcolormesh(rx[::stepx], ry[::stepy], v[::stepy, ::stepx], vmin=vmin, vmax=vmax, norm=norm, cmap=cmap)
        if topography is not None:
            ax.contour(rx[::stepx], ry[::stepy], topography[::stepy, ::stepx], 8, cmap='gist_earth', linewidths=0.5)
        ax.set_aspect(aspect)
        ax.invert_yaxis()
        if sx > 0 and sy > 0:
            ax.scatter(sx, sy, 150, color='g', edgecolor='k', marker='*')
            for dx in (-25, 0, 25, 50, 75, 100):
                for dy in (-50, -25, 0, 25, 50, 75):
                    ax.scatter(sx + dx * dh[0], sy + dy * dh[0], 20, color='g', edgecolor='k', marker='^')
        for i in range(1, NPX):
            ax.axvline(cfg.x[0] // i * dh[0], linestyle=':', linewidth=1.2, color='c')
        for i in range(1, NPY):
            ax.axhline(cfg.y[0] // i * dh[0], linestyle=':', linewidth=1.2, color='c')
        
        title = f'T = {t:.3f}s'
        xlabel = xlabel or f'X ({space_scale})'
        ylabel = ylabel or f'Y ({space_scale})'
        cbar_label = f'V{comp} ({unit})'
        self.help_plot_slice(fig, ax, im, aspect, title, xlabel, ylabel, cbar_label, orientation, fig_name, tick_limit=1e-3)
       
        # For la habra specificaly, plotting coast line
        if lahabra:
            coastline = np.genfromtxt('coastline.idx', dtype='int', usecols=[1,2])
            ax.plot(coastline[:, 0] * dh[0], coastline[:, 1] * dh[0], 'k')
        return save_image(fig)


