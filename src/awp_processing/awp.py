'''The main analyzing subroutine
TODO:
    1. Use multiprocessing to accelerate?
    2. Add plot subroutines, maybe in a seperate .py
'''
from pathlib import Path
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from collections import defaultdict
from obspy.signal.detrend import polynomial
import sys

from .utils import *
from .read_params import read_params
from .filter_BU import filt_B


class Scenario:
    def __init__(self, model="", case="", conf_file='param.sh'):
        '''
        Input:
            case (str): different cases of a model, if exists
        '''
        self.output_dir = Path(model, "output_sfc" if not case else f"{case}")
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
        '''
        if i < 0:
            return int(i)
        bg = self.cfg['nbg' + comp][block]
        ed = self.cfg['ned' + comp][block]
        skip = self.cfg['nskp' + comp][block]
        if i < bg or i > ed:
            print(f"i{comp}: outside output range. Rounded")
            i = min(max(i, bg), ed)
        shift = i - bg
        if shift % skip != 0:
            print(f"i{comp}: rounded using nearest grid")
        return int(round(shift / skip))


    def reindex_block(self, ix, iy, iz=1, block=0):
        '''Shift and compress index, return 0-indexed'''
        ix = self.reindex(ix, 'x')
        iy = self.reindex(iy, 'y')
        iz = self.reindex(iz, 'z')
        return ix, iy, iz
    
    
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
        nt = int(self.cfg.tmax / self.cfg.dt)
        nx, ny, nz = self.cfg.nx[block], self.cfg.ny[block], self.cfg.nz[block]
        ix, iy, iz = self.reindex_block(ix, iy, iz, block=block)
        print(f"\nShape of velocity output: ({nz}, {ny}, {nx})")

        resi, it = np.divmod(int(t / self.cfg.dt / self.cfg.tskip), self.cfg.wstep) 
        fnum = (resi + 1) * self.cfg.wstep * self.cfg.tskip 
        file_name = f'{self.output_dir}/S{comp}_{block}_{fnum:07d}'
        print(f'\rt = {t}s / {self.cfg.tmax}s, {it * self.cfg.tskip} / {fnum}, file = {file_name}', end="\r", flush=True)
        
        v = np.fromfile(file_name, dtype='float32', count=nz * ny * nx,
                        offset=it * nx * ny * nz * 4).reshape(nz, ny, nx)
        idx = np.where(np.isnan(v))
        if np.isnan(v).any():
            print(f"\n{len(idx[0])} NANs founded\n")

        if ix >= 0:
            v = v[:, :, ix]
            print(f'\nThe x_index is: {ix * self.cfg.nskpx[block]} / {self.cfg.nedx[block]}, Vmax = {np.max(v)}')
        elif iy >= 0:
            v = v[:, iy, :]
            print(f'\nThe y_index is: {iy * self.cfg.nskpy[block]} / {self.cfg.nedy[block]}, Vmax = {np.max(v)}')
        elif iz >= 0:
            v = v[iz, :, :]
            print(f'\nThe z_index is: {iz * self.cfg.nskpz[block]} / {self.cfg.nedz[block]}, Vmax = {np.max(v)}')
        return v.copy()
    
    
    def read_syn(self, ix, iy, iz=1, block=0, comps='xyz'):
        '''Read synthetics
        Input:
            ix, iy, iz (int): Indices of site in the original mesh, 1-indexed
        Output:
            v (record array): three-component velocities and dt
        '''
        import struct
        
        nx, ny, nz = self.cfg.nx[block], self.cfg.ny[block], self.cfg.nz[block]
        ix, iy, iz = self.reindex_block(ix, iy, iz, block=block)
        
        v = defaultdict(list)
        print(f"ix={ix}, iy={iy}, iz={iz}")
        skips = [4 * (j * nz * ny * nx + 
                      iz * ny * nx + 
                      iy * nx + ix) for j in range(self.cfg.wstep)]
        dt = self.cfg.dt * self.cfg.tskip 
        nfile = int(self.cfg.tmax / dt) // self.cfg.wstep
        file_step = self.cfg.wstep * self.cfg.tskip
        for comp in comps:
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

    def preproc_vel(self, v, lowf=0.5, highf=12):
        '''Pre-processing velocities
        Input:
            v (dict): velocities, 3-components and dt
            highf (float): highcut frequency for filtering
        '''
        from scipy.signal import detrend
        for c in 'xyz':
            v[c] = v[c] - np.mean(v[c])
            v[c] = detrend(v[c].astype(np.float64), overwrite_data=True)
            v[c] = filt_B(v[c], 1 / v['dt'], lowcut=lowf, highcut=highf, order=5)
            disp = np.cumsum(v[c]) * v['dt']
            disp = polynomial(disp, order=2)
            v[c] = np.diff(disp, prepend=0) / v['dt']
        return v


    def comp_spec(self, ix, iy, iz=1, fmax=0, block=0, lowf=0.5, highf=12, comps='xy', metric='vel', output_vel=False):
        '''Compute spectrum of velocity / acceleration
        Input:
            ix, iy, iz (int): Indices of site in the original mesh, 1-indexed
            fmax (float): Maximum frequency to store outputs
            block (int): # of block in DM code
            highf (float): highcut frequency for filtering

        Output:
            fv (dict): three components spectrum and frequency
        '''
        v = self.read_syn(ix, iy, iz, block)
        v = self.preproc_vel(v, lowf=lowf, highf=highf)
        fs = 1 / v['dt']
        df = fs / len(v['x'])
        fmax = fmax or min(highf, fs / 2)  # if not define fmax, choose highf or f_nyquist
        fv = AttrDict()
        fv['f'] = np.arange(df, fmax + df, df)
        for c in comps:
            if metric == 'acc':
                v[c] = np.diff(v[c], prepend=0) / v['dt']  
            fv[c] = np.abs(np.fft.fft(v[c]) * v['dt'] * 2)[:len(fv['f'])]
        if output_vel:
            return v, fv
        return fv


    def plot_slice_x(self, t, ix=0, sy=0, sz=0, vmin=None, vmax=None, block=0, NPY=1, comp='Y', norm='log', unit='km', fig_name=None):
        """Plot snapshot at certain x
        Input
        -----
        t : float
            time of snapshot
        ix : int
            location of snapshot
        sy, sz : float
            location of source
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
        unit : ['m', 'km']
            Unit of axis, to avoid too many digits
        fig_name : None or string 
            If string, then save image to file
        """

        cfg = self.cfg
        mesh_bot = 0  # the bottom the each mesh
        if norm == 'log':
            norm = colors.SymLogNorm(linthresh=1e-4, base=10)
        else:
            norm = None
        
        fig, ax = plt.subplots(dpi=400)
        for i in range(block + 1):
            ix = (ix - 1) // (3 ** i) + 1
            v = self.read_slice(t, ix=ix, block=block, comp=comp)
            vmin = 1.2 * min(np.min(v)) if i == 0 and vmin is None else vmin
            vmax = 0.8 * max(np.max(v)) if i == 0 and vmax is None else vmax
                
            ry = np.arange(cfg.nbgy[block] - 1, cfg.nedy[block], cfg.nskpy[block]) * cfg.dh[block]
            rz = np.arange(cfg.nbgz[block] - 1, cfg.nedz[block], cfg.nskpz[block]) * cfg.dh[block]
            if unit == 'km':
                sx, sy = sx / 1000, sy / 1000
                rx, ry = rx / 1000, ry / 1000
            stepy = decimate(ry)
            stepz = decimate(rz)
            im = ax.pcolormesh(ry[::stepy], rz[::stepz] + mesh_bot, v[::stepy, ::stepz], vmin=vmin, vmax=vmax, norm=norm, cmap='coolwarm')
            mesh_bot += (cfg.z[block] - 7) * cfg.dh[block]
            print(f'Mesh bottom of block {block} is = {mesh_bot}')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.scatter(sx, sy, 150, color='g', edgecolor='k', marker='*')
        for i in range(1, NPY):
            ax.axvline(cfg.y[0] // i * cfg.dh[0], linestyle=':', linewidth=1.2, color='c')
        ax.set(xlabel=f'Y ({unit})', ylabel=f'Z ({unit})', title=f'T = {t:.1f}s')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'V{comp} (m/s)')
        if fig_name:
            fig.savefig(f"{fig_name}", dpi=600, bbox_inches='tight', pad_inches=0.05)
        return save_image(fig)


    def plot_slice_y(self, t, iy=0, sx=0, sz=0, vmin=None, vmax=None, block=0, NPX=1, comp='X', norm='log', unit='km', fig_name=None):
        """Plot snapshot at certain x
        Input
        -----
        t : float
            time of snapshot
        iy : int
            location of snapshot
        sx, sz : float
            location of source
        vmin, vmax : float
            range of velocity to plot
        block : int
            the deepest block to plot
        NPX : int
            Partition of GPUs along y
        comp : ['X', 'Y', 'Z']
            Component of velocity
        norm : 'log' or else
            If 'log', apply log scale to image
        unit : ['m', 'km']
            Unit of axis, to avoid too many digits
        fig_name : None or string 
            If string, then save image to file
        """

        cfg = self.cfg
        mesh_bot = 0  # the bottom the each mesh
        if norm == 'log':
            norm = colors.SymLogNorm(linthresh=1e-4, base=10)
        else:
            norm = None
        fig, ax = plt.subplots(dpi=400)
        for i in range(block + 1):
            iy = (iy - 1) // (3 ** i) + 1
            v = self.read_slice(t, iy=iy, block=block, comp=comp)
            vmin = 1.2 * min(np.min(v)) if i == 0 and vmin is None else vmin
            vmax = 0.8 * max(np.max(v)) if i == 0 and vmax is None else vmax
                
            rx = np.arange(cfg.nbgx[block] - 1, cfg.nedx[block], cfg.nskpx[block]) * cfg.dh[block]
            rz = np.arange(cfg.nbgz[block] - 1, cfg.nedz[block], cfg.nskpz[block]) * cfg.dh[block]
            if unit == 'km':
                sx, sy = sx / 1000, sy / 1000
                rx, ry = rx / 1000, ry / 1000
            stepx = decimate(rx)
            stepz = decimate(rz)
            im = ax.pcolormesh(rx[::stepx], rz[::stepz] + mesh_bot, v[::stepx, ::stepz], vmin=vmin, vmax=vmax, norm=norm, cmap='coolwarm')
            mesh_bot += (cfg.z[block] - 7) * cfg.dh[block]
            print(f'Mesh bottom of block {block} is = {mesh_bot}')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.scatter(sx, sy, 150, color='g', edgecolor='k', marker='*')
        for i in range(1, NPX):
            ax.axvline(cfg.x[0] // i * cfg.dh[0], linestyle=':', linewidth=1.2, color='c')
        ax.set(xlabel=f'X ({unit})', ylabel=f'Z ({unit})', title=f'T = {t:.1f}s')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'V{comp} (m/s)')
        if fig_name:
            fig.savefig(f"{fig_name}", dpi=600, bbox_inches='tight', pad_inches=0.05)
        return save_image(fig)


    def plot_slice_z(self, t, iz=0, sx=0, sy=0, vmin=None, vmax=None, block=0, NPX=1, NPY=1, comp='X', unit='km', norm='log', fig_name=None):
        """Plot snapshot at certain x
        Input
        -----
        t : float
            time of snapshot
        iz : int
            location of snapshot
        sx, sy : float
            location of source
        vmin, vmax : float
            range of velocity to plot
        block : int
            the deepest block to plot
        NPX, NPY : int
            Partition of GPUs along y
        comp : ['X', 'Y', 'Z']
            Component of velocity
        unit : ['m', 'km']
            Unit of axis, to avoid too many digits
        norm : 'log' or else
            If 'log', apply log scale to image
        fig_name : None or string 
            If string, then save image to file
        """

        cfg = self.cfg
        if norm == 'log':
            norm = colors.SymLogNorm(linthresh=1e-4, base=10)
        else:
            norm = None
        fig, ax = plt.subplots(dpi=400)
        
        v = self.read_slice(t, iz=iz, block=block, comp=comp)
        vmin = 1.2 * min(np.min(v)) if vmin is None else vmin
        vmax = 0.8 * max(np.max(v)) if vmax is None else vmax
            
        rx = np.arange(cfg.nbgx[block] - 1, cfg.nedx[block], cfg.nskpx[block]) * cfg.dh[block]
        ry = np.arange(cfg.nbgy[block] - 1, cfg.nedy[block], cfg.nskpy[block]) * cfg.dh[block]
        if unit == 'km':
            sx, sy = sx / 1000, sy / 1000
            rx, ry = rx / 1000, ry / 1000
        stepx = decimate(rx)
        stepy = decimate(ry)
        im = ax.pcolormesh(rx[::stepx], ry[::stepy], v[::stepy, ::stepx], vmin=vmin, vmax=vmax, norm=norm, cmap='coolwarm')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.scatter(sx, sy, 150, color='g', edgecolor='k', marker='*')
        for i in range(1, NPX):
            ax.axvline(cfg.x[0] // i * cfg.dh[0], linestyle=':', linewidth=1.2, color='c')
        for i in range(1, NPY):
            ix.axhline(cfg.y[0] // i * cfg.dh[0], linestyle=':', linewidth=1.2, color='c')
        ax.set(xlabel=f'X ({unit})', ylabel=f'Y ({unit})', title=f'T = {t:.1f}s')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'V{comp} (m/s)')
        if fig_name:
            fig.savefig(f"{fig_name}", dpi=600, bbox_inches='tight', pad_inches=0.05)
        return save_image(fig)
