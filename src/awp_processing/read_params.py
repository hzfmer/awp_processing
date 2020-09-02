''' Read parameters in param.sh
'''

import argparse
import re
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

    parser.add_argument('-X', type=int, help='Number of X-grids in the lowest block')
    parser.add_argument('-Y', type=int, help='Number of Y-grids in the lowest block')
    parser.add_argument('-Z', type=split_arg, help='List of number of z-grids from top to bottom')
    parser.add_argument('-G', type=int, default=1, help='Number of blocks')
    parser.add_argument('--DH', type=float,  help='Grid spacing in the bottom block')
    parser.add_argument('--NBGX', type=split_arg)
    parser.add_argument('--NBGY', type=split_arg)
    parser.add_argument('--NBGZ', type=split_arg)
    parser.add_argument('--NEDX', type=split_arg)
    parser.add_argument('--NEDY', type=split_arg)
    parser.add_argument('--NEDZ', type=split_arg)
    parser.add_argument('--NSKPX', type=split_arg, default=0)
    parser.add_argument('--NSKPY', type=split_arg, default=0)
    parser.add_argument('--NSKPZ', type=split_arg, default=0)
    parser.add_argument('--DT', type=float)
    parser.add_argument('--TMAX', type=float)
    parser.add_argument('--NSRC', type=split_arg, help='Number of sources from top to bottom')
    parser.add_argument('--NST', type=int)
    parser.add_argument('--NTISKP', type=int, dest='tskip', help='Step to skip when generating outputs')
    parser.add_argument('--NVAR', type=int)
    parser.add_argument('--SXRGO', default=repr('output_sfc/SX_0_'))
    parser.add_argument('--SYRGO', default=repr('output_sfc/SY_0_'))
    parser.add_argument('--SZRGO', default=repr('output_sfc/SZ_0_'))
    parser.add_argument('-o', default=repr('output_sfc'))
    parser.add_argument('--READ_STEP', type=int, help='Steps in each batch to read the source')
    parser.add_argument('--WRITE_STEP', type=int, dest='wstep', help='Steps to write in a single output file')
    parser.add_argument('--IVELOCITY', type=int, default=0, help='Aggregative = 1, otherwise 0')

    args = utils.AttrDict(vars(parser.parse_known_args([f'@{f_param}'])[0]))

    # Convert some scalars to vectors
    args['ratio'] = args['z'][0] // args['z'][1] if args['g'] > 1 else 1
    for key in ['x', 'y']:
        args[key] = ext_block(args[key], args['g'], ratio=args['ratio'])
    args['dh'] = ext_block(args['dh'], args['g'], ratio=1/args['ratio'])
    
    # If not specified, force to use 1s in skips in every direction
    for c in 'xyz':
        if not args[f'nskp{c}'] or len(args[f'nskp{c}']) != args['g']:      
            args[f'nskp{c}'] = [1] * args['g']

    return args

