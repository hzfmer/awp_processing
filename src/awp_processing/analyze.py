'''The main analyzing subroutine
TODO:
    1. Use multiprocessing to accelerate?
    2. Add plot subroutines, maybe in a seperate .py
'''
from pathlib import Path
import numpy as np
from collections import defaultdict
from obspy.signal.detrend import polynomial

from .utils import AttrDict
from .read_params import read_params
from .filter_BU import filt_B


class Scenario():
    def __init__(self, model="", case="", f_param='param.sh'):
        '''
        Input:
            case (str): different cases of a model, if exists
        '''
        self.output_dir = Path(model, "output_sfc" if not case else f"output_sfc_{case}")
        self.cfg = AttrDict(read_params(Path(model, f_param)))
        self.shape_of_block()
        
 
    def shape_of_block(self):
        '''Shape of block, (nx, ny, nz)'''
        self.cfg.nx = [0] * self.cfg.g
        self.cfg.ny = [0] * self.cfg.g
        self.cfg.nz = [0] * self.cfg.g
        print(self.cfg.g)
        for g in range(self.cfg.g):
            print(type(self.cfg.nedx), type(self.cfg.nskpx), self.cfg.nskpx)
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
    
    
    def read_slice(self, it, ix=-1, iy=-1, iz=-1, block=0, comp="X"):
        '''Read wave field snapshot
        Input:
            it: time step to read
            ix, iy, iz (int): Indices of cross-section in the original mesh, 1-indexed
            block: The index of the block, 0-indexed
            comp: different cases of a model, if exists
        '''
        
        nt = int(self.cfg.tmax / self.cfg.dt)
        nx, ny, nz = self.cfg.nx[block], self.cfg.ny[block], self.cfg.nz[block]
        ix, iy, iz = self.reindex_block(ix, iy, iz, block=block)
        print(f"\nShape of velocity output: ({nz}, {ny}, {nx})")
        
        fnum = int(np.ceil((it + 1) / self.cfg.wstep) * self.cfg.wstep * self.cfg.tskip)
        file_name = f'{self.output_dir}/S{comp}_{block}_{fnum:07d}'
        print(f'\r{it} / {nt}, t = {self.cfg.dt * it}s, file = {file_name}', end="\r", flush=True)
        
        v = np.fromfile(file_name, dtype='float32', count=nz * ny * nx,
                        offset=it * nx * ny * nz * 4).reshape(nz, ny, nx)
        print(v.shape)
        print(ix, iy, iz)
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
        return v.copy(), idx
    
    
    def read_syn(self, ix, iy, iz=1, block=0):
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
        for comp in 'xyz':
            for i in range(1, nfile + 1):
                with open(f'{self.output_dir}/S{comp.upper()}_{block}_{i * file_step:07d}', 'rb') as fid:
                    for j in range(self.cfg.wstep):
                        fid.seek(skips[j], 0)
                        v[comp] += struct.unpack('f', fid.read(4))
                        if np.isnan(v[comp]).any():
                            print(f"\nNAN in file {self.output_dir}/S{comp.upper()}_{block}_{i * file_step}\n")
                            return None
            v[comp] = np.array(v[comp])
        # v['X'] = -v['X'], for la habra validation only
        v['dt'] = dt
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
