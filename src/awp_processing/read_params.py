''' Read parameters in param.sh
'''

import argparse
import re
from pathlib import Path
import numpy as np
from . import utils 

def _convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        if arg[0] == '#':
            break
        yield arg
        

def split_arg(string):
    """Split multiple input args, seperated by ","
    """
    return list(map(int, re.findall(r'\d+\.*\d*', string))) if ',' in string else [int(string)]
#    return list(map(int, map( string.split(','))) if ',' in string else [int(string)]


def to_list(string):
    return [float(string)]


def read_coords(file_name, base, prefix=""):
    """Read Ossian's input files with prefix indicating type"""
    res = {}
    with open(Path(base, file_name), 'r') as fid:
        nline = 0
        for line in fid:
            nline += 1
            if "=" in line:
                k, v = line.strip('\n').split("=")
                res[prefix + "_" + k] = int(v) if v.isdigit() else Path(base, v) 
            if "coordinates" in line:
                break
    res[prefix + "_coords"] = np.genfromtxt(Path(base, file_name), skip_header=nline, dtype='float32')[:, 1:]
    return res 

def ext_block(var, nblocks, ratio=3):
    """
    Extend some variables defined in the bottom block only

    Input
    -----
    var: float or int
        Variable in the bottom block
    nblocks: int
        Number of blocks, often 2
    ratio: float
        ratio between upper block to the block below

    Return
    ------
    res : list
        Variable in all blocks, from top to bottom
    """
    return [var * (ratio ** i) for i in range(nblocks)[::-1]]

def read_params(f_param):
    parser = argparse.ArgumentParser(description="Read the parameters in f_param, output the dict containing the options and values", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = _convert_arg_line_to_args

    parser.add_argument('--NX', '-X', type=int, help='Number of X-grids in the lowest block')
    parser.add_argument('--NY', '-Y', type=int, help='Number of Y-grids in the lowest block')
    parser.add_argument('--NZ', '-Z', type=split_arg, help='List of number of z-grids from top to bottom')
    parser.add_argument('--PX', '-x', type=int, help='Number of CPU along x direction')
    parser.add_argument('--PY', '-y', type=int, help='Number of CPU along y direction')
    parser.add_argument('--PZ', '-z', type=int, default=1, help='Number of CPU along z direction')
    parser.add_argument('-G', type=int, default=1, help='Number of blocks')
    parser.add_argument('--DH', '-H', type=float,  help='Grid spacing in the bottom block')
    parser.add_argument('--ND', '-D', type=int, help='Number of ABC layers')
    parser.add_argument('--NBGX', type=split_arg)
    parser.add_argument('--NBGY', type=split_arg)
    parser.add_argument('--NBGZ', type=split_arg)
    parser.add_argument('--NEDX', type=split_arg)
    parser.add_argument('--NEDY', type=split_arg)
    parser.add_argument('--NEDZ', type=split_arg)
    parser.add_argument('--NSKPX', type=split_arg, default=0)
    parser.add_argument('--NSKPY', type=split_arg, default=0)
    parser.add_argument('--NSKPZ', type=split_arg, default=0)
    parser.add_argument('--DT', '-t', type=float)
    parser.add_argument('--TMAX', '-T', type=float)
    parser.add_argument('--NSRC', '-S', type=split_arg, help='Number of sources from top to bottom')
    parser.add_argument('--NST', '-N', type=int)
    parser.add_argument('--NTISKP', type=int, dest='tskip', help='Step to skip when generating outputs')
    parser.add_argument('--NVAR', type=int, help='Nubmer of variables in the mesh [3, 5, 8]')
    parser.add_argument('--IVELOCITY', type=int, default=0, help='Aggregative = 1, otherwise 0')
    parser.add_argument('--READ_STEP', '-R', type=int, help='Steps in each batch to read the source')
    parser.add_argument('--READ_STEP_GPU', '-Q', type=int, default=1, help='CPU reads larger chunks and send to GPU every READ_STEP_GPU steps')
    parser.add_argument('--WRITE_STEP', '-W', type=int, dest='wstep', help='Steps to write in a single output file')
    parser.add_argument('--SXRGO', default=repr('output_sfc/SX_0_'))
    parser.add_argument('--SYRGO', default=repr('output_sfc/SY_0_'))
    parser.add_argument('--SZRGO', default=repr('output_sfc/SZ_0_'))
    parser.add_argument('--INSRC', default="source", help='eg. source_0, source_1')
    parser.add_argument('--INVEL', default="mesh", help='eg. mesh_0, mesh_1')
    parser.add_argument('--INTOPO', help='header: (nx, ny, pad_length), shape=(nx, ny)')
    parser.add_argument('--CHKFILE', '-c', default=repr('output_ckp/ckp'))
    parser.add_argument('--OUT', '-o', default=repr('output_sfc'))
    parser.add_argument('--SOURCEFILE', default="", help="Source input file that uses coordinates instead of indices to specify the position")
    parser.add_argument('--SGTFILE', default="", help="Strain Green's tensor output file")
    parser.add_argument('--FORCEFILE', default="", help="Boundary point force input file")
    parser.add_argument('--RECVFILE', default="", help='Receiver output file')
    args = utils.AttrDict(vars(parser.parse_known_args([f'@{f_param}'])[0]))

    # Additional parameters
    base = Path(f_param).parent
    if args['recvfile']:
        args.update(read_coords(args['recvfile'], base, prefix="recv"))
    if args['sourcefile']:
        args.update(read_coords(args['sourcefile'], base, prefix="src"))
    if args['sgtfile']:
        args.update(read_coords(args['sgtfile'], base, prefix="sgt"))
    if args['forcefile']:
        args.update(read_coords(args['forcefile'], base, prefix="force"))

    # Convert some scalars to vectors
    args['ratio'] = 3
    for key in ['x', 'y', 'z']:
        if key not in args:
            args[key] = args['n' + key]
        if key != 'z':
            args[key] = args['n' + key] = ext_block(args[key], args['g'], ratio=args['ratio'])
    args['dh'] = ext_block(args['dh'], args['g'], ratio=1/args['ratio'])
    args['nt'] = int(args['tmax'] / args['tskip'] / args['dt'])
    
    # If not specified, force to use 1s in skips in every direction
    for c in 'xyz':
        if not args[f'nskp{c}'] or len(args[f'nskp{c}']) != args['g']:      
            args[f'nskp{c}'] = [1] * args['g']

    return args

