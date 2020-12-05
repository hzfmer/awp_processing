import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def read_mesh(fmesh, nx, ny, nz, ix=-1, iy=-1, iz=-1, nvar=3, ivar=1, nbit=4):
    """
    Input
    -----
    ivar : int 
        Normally (vp, vs, rho, vp/vs)
    nbit : int 
        Size of datatype (4 for float32)
    """
    count = nx * ny * nvar
    off_layer = nbit * count # single float only
    if ix >= 0:
        length = ny * nz
    elif iy >= 0: 
        length = nx * nz 
    else:
        length = nx * ny

    res = np.zeros(length, dtype='float32')
    with open(fmesh, 'rb') as fid:
        if ix >= 0:
            for i in range(nz):
                tmp = np.fromfile(fid, count=count,
                        dtype='float32').reshape(ny, nx, nvar)
                if ivar == 3:
                    tmp = tmp[:, :, 0] / tmp[:, :, 1]
                else:
                    tmp = tmp[:, :, ivar]
                res[i * ny : (i + 1) * ny] = tmp[:, ix]
        elif iy >= 0:
            for i in range(nz):
                tmp = np.fromfile(fid, count=count,
                        dtype='float32').reshape(ny, nx, nvar)
                if ivar == 3:
                    tmp = tmp[:, :, 0] / tmp[:, :, 1]
                else:
                    tmp = tmp[:, :, ivar]
                res[i * nx : (i + 1) * nx] = tmp[iy, :]
        else:
            tmp = np.fromfile(fid, count=count, offset=off_layer * iz,
                    dtype='float32').reshape(ny, nx, nvar)
            if ivar == 3:
                tmp = tmp[:, :, 0] / tmp[:, :, 1]
            else:
                tmp = tmp[:, :, ivar]

            return tmp
    if ix >= 0:
        return np.reshape(res, (nz, ny))
    else:
        return np.reshape(res, (nz, nx))


def plot_mesh(fmesh, nx, ny, nz, dh=1, unit='km', ix=-1, iy=-1, iz=-1, nvar=3, ivar=1, step1=2, step2=2, cmap='inferno', xlabel=None, ylabel=None, save=False, file_name=None):
    mesh = read_mesh(fmesh, nx, ny, nz, ix, iy, iz, nvar=nvar, ivar=ivar)
    print(f'Max = {mesh.max():.2f}, Min = {mesh.min():.2f}')
    cbar_labels = {0: "Vp (m/s)", 1: "Vs (m/s)", 2: "Density (m/s)", 3: "Vp / Vs"}
    if ix >= 0:
        xxlabel, yylabel = 'Y', 'Z'
        xs = np.arange(ny) * dh 
        ys = np.arange(nz) * dh
        orient = f'X_{ix}'
    elif iy >= 0:
        xxlabel, yylabel = 'X', 'Z'
        xs = np.arange(nx) * dh 
        ys = np.arange(nz) * dh
        orient = f'Y_{iy}'
    else:
        xxlabel, yylabel = 'X', 'Y'
        xs = np.arange(nx) * dh 
        ys = np.arange(ny) * dh
        orient = f'Z_{iz}'

    xlabel = xlabel or xxlabel
    ylabel = ylabel or yylabel

    fig, ax = plt.subplots(dpi=500)
    im = ax.pcolormesh(xs[::step1], ys[::step2], mesh[::step2, ::step1], cmap=cmap,
            rasterized=True) 
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if not iz >= 0:
        ax.invert_yaxis()
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(f'{cbar_labels[ivar]}')
    if save:
        if not file_name:
            file_name = f'{fmesh.split(".")[0]}_slice_{orient}.pdf'
        fig.savefig(f"results/{file_name}.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(f"results/{file_name}", dpi=600, bbox_inches='tight', pad_inches=0.05)
    return fig


def check_mesh_cont(fmesh_0, fmesh_1, nx, ny, nz, nvar=3, skip=3, verbose=True, plot=False):
    """
    Check continuity of meshes for AWP-DM

    Input
    -----
    fmesh_0 : str
        Filename for top mesh 
    fmesh_1 : str
        Filename for bottom mesh 
    nx, ny, nz : int
        Shape of the top mesh 
    nvar : int 
        Number of raviables stored in mesh 
    skip : int 
        Ratio of increasing spacial griding in AWP-DM 
    verbose : bool 
        Print more information 
    plot : bool 
        Plot diff between meshes
    """
    max_diff = 0
    with open(fmesh_0, 'rb') as f0, open(fmesh_1, 'rb') as f1:
        f0.seek(4 * nvar * nx * ny * (nz - 8), 0)
        for i in range(3):       
            data0 = np.frombuffer(f0.read(4 * nvar * nx * ny),
                                  dtype='float32').reshape(ny, nx, nvar)
            data1 = np.frombuffer(f1.read(4 * nvar * nx * ny // skip ** 2),
                                  dtype='float32').reshape(ny//skip, nx//skip, nvar)
            diff = data0[1::skip, ::skip, :] - data1
            max_diff = max(np.max(diff), max_diff)
            #print(data0[:5, :5, 1], data1[:2, :2, 1])
            if not np.isclose(diff, 0).all():
                loc_y, loc_x, loc_z = np.unravel_index(
                                      np.argmax(np.abs(diff)), diff.shape)
                if verbose:
                    print(f"Not consistent, max difference at {np.argmin(diff)}: {np.min(diff)}")
                    print(f"({loc_y}, {loc_x}, {loc_z}) in {diff.shape}")
                    print("Top block: ", data0[1 + skip * loc_y, skip * loc_x, loc_z])
                    print("Bottom block: ", data1[loc_y, loc_x, loc_z])
            f0.seek(4 * nvar * nx * ny * 2, 1)
    if np.isclose(0, max_diff):
        print("Top and bottom blocks are consistent!")   
    elif plot:
        im=plt.imshow(diff[:, :, loc_z], cmap='RdBu')
        plt.colorbar(im)
        plt.savefig(f"temp_mesh_diffcont.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
        print(f"Top and bottom blocks are not consistent! Max_diff = {max_diff}")
    return True

