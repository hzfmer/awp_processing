''' Read parameters in param.sh
'''

import argparse
import utils


def _convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        if arg[0] == '#':
            break
        yield arg
        

def split_arg(string):
    return list(map(int, string.split(','))) if ',' in string else int(string)


def read_params(f_param):
    parser = argparse.ArgumentParser(description="Read the parameters in f_param, output the dict containing the options and values", fromfile_prefix_chars="@")
    parser.convert_arg_line_to_args = _convert_arg_line_to_args

    parser.add_argument('-X', type=int, help='Number of X-grids in the lowest block')
    parser.add_argument('-Y', type=int, help='Number of Y-grids in the lowest block')
    parser.add_argument('-Z', type=split_arg, help='List of number of z-grids from top to bottom')
    parser.add_argument('-G', type=int, default=1, help='Number of blocks')
    parser.add_argument('--NBGX', type=split_arg)
    parser.add_argument('--NBGY', type=split_arg)
    parser.add_argument('--NBGZ', type=split_arg)
    parser.add_argument('--NEDX', type=split_arg)
    parser.add_argument('--NEDY', type=split_arg)
    parser.add_argument('--NEDZ', type=split_arg)
    parser.add_argument('--NSKPX', type=split_arg, default=1)
    parser.add_argument('--NSKPY', type=split_arg, default=1)
    parser.add_argument('--NSKPZ', type=split_arg, default=1)
    parser.add_argument('--DT', type=float)
    parser.add_argument('--TMAX', type=float)
    parser.add_argument('--NSRC', type=split_arg, help='Number of sources from top to bottom')
    parser.add_argument('--NST', type=int)
    parser.add_argument('--NTISKP', type=int, help='Step to skip when generating outputs')
    parser.add_argument('--NVAR', type=int)
    parser.add_argument('--SXRGO', default=repr('output_sfc/SX_0_'))
    parser.add_argument('--SYRGO', default=repr('output_sfc/SY_0_'))
    parser.add_argument('--SZRGO', default=repr('output_sfc/SZ_0_'))
    parser.add_argument('-o', default=repr('output_sfc'))
    parser.add_argument('--READ_STEP', type=int, help='Steps in each batch to read the source')
    parser.add_argument('--WRITE_STEP', type=int, help='Steps to write in a single output file')
    parser.add_argument('--IVELOCITY', type=int, default=0, help='Aggregative = 1, otherwise 0')

    args = utils.AttrDict(vars(parser.parse_known_args([f'@{f_param}'])[0]))
    if args['G'] > 1:
        # If not specified, force to use 1s in skips in every direction
        for c in 'XYZ':
            if type(args[f'NSKP{c}']) != list:      
                args[f'NSKP{c}'] = [1] * args['G']

    return args

