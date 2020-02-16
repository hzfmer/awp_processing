from pathlib import Path
import numpy as np
from collections import defaultdict
from awp_processing import utils
from .read_params import read_params
from .filter_BU import filt_B


class output():
    def __init__(self, model="", case="", f_param='param.sh'):
        '''
        Input:
            case (str): different cases of a model, if exists
        '''
        self.output_dir = Path(model, "output_sfc" if not case else f"output_sfc_{case}")
        self.cfg = utils.AttrDict(read_params(Path(model, f_param)))
        self.shape_of_block()
        
 
    def shape_of_block(self):
        '''Shape of block, (nx, ny, nz)'''
        self.cfg.nx = [[] * self.cfg.g]
        self.cfg.ny = [[] * self.cfg.g]
        self.cfg.nz = [[] * self.cfg.g]
        for g in self.cfg.g:
            self.cfg.nx[g] = (self.cfg.endx[g] - self.cfg.bgx[g]) // self.cfg.skpx[g] + 1
            self.cfg.ny[g] = (self.cfg.endy[g] - self.cfg.bgy[g]) // self.cfg.skpy[g] + 1
            self.cfg.nz[g] = (self.cfg.endz[g] - self.cfg.bgz[g]) // self.cfg.skpz[g] + 1
        
    
    def reindex_block(self, ix=0, iy=0, iz=0, block=0):
        '''Shift and compress index, 0-indexed'''
        ix = int((ix - self.cfg.bgx[block]) / self.cfg.skpx[block])
        iy = int((iy - self.cfg.bgy[block]) / self.cfg.skpy[block])
        iz = int((iz - self.cfg.bgz[block]) / self.cfg.skpz[block])
        return (x for x in (ix, iy, iz) if x >= 0)
    
    
    def read_slice(self, it, ix=0, iy=0, iz=0, block=0, comp="X"):
        '''Read wave field snapshot
        Input:
            it: time step to read
            ix, iy, iz (int): Indices of cross-section, 1-indexed
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
        
        idx = np.where(np.isnan(v))
        if np.isnan(v).any():
            print(f"\n{len(idx[0])} NANs founded\n")

        if ix >= 0:
            v = v[:, :, ix]
            print(f'\nThe x_index is: {ix * self.cfg.skpx[block]} / {self.cfg.endx[block]}, Vmax = {np.max(v)}')
        elif iy > 0:
            v = v[:, iy, :]
            print(f'\nThe y_index is: {iy * self.cfg.skpy[block]} / {self.cfg.endy[block]}, Vmax = {np.max(v)}')
        elif iz >= 0:
            v = v[iz, :, :]
            print(f'\nThe z_index is: {iz * self.cfg.skpz[block]} / {self.cfg.endz[block]}, Vmax = {np.max(v)}')
        return v.copy(), idx
    
    
    def read_syn(self, ix, iy, iz=0, block=0):
        '''Read synthetics
        Input:
            ix, iy, iz (int): Indices of site, 1-indexed
        '''
        import struct
        
        nx, ny, nz = self.cfg.nx[block], self.cfg.ny[block], self.cfg.nz[block]
        ix, iy, iz = self.reindex_block(ix, iy, iz, block=block)
        
        v = defaultdict(list)
        skips = [4 * (j * nz * ny * nx + 
                      iz * ny * nx + 
                      iy * nx + ix) for j in range(self.cfg.wstep)]
        nfile = int(self.cfg.tmax / self.cfg.dt) // self.cfg.wstep
        file_step = self.cfg.wstep * self.cfg.tskip
        for comp in 'XYZ':
            for i in range(1, nfile + 1):
                with open(f'{self.output_dir}/S{comp}_{block}_{i * file_step:07d}', 'rb') as fid:
                    for j in range(self.cfg.wstep):
                        fid.seek(skips[j], 0)
                        v[comp] += struct.unpack('f', fid.read(4))
                        if np.isnan(v[comp]).any():
                            print(f"\nNAN in file {self.output_dir}/S{comp}_{block}_{(i - 1) * file_step}\n")
                            return None
                        v[comp] = np.array(v[comp])
        v['X'] = -v['X']
        v['dt'] = self.cfg.dt
        return v
