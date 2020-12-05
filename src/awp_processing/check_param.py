import argparse
from collections.abc import Iterable
import numpy as np
from awp_processing import awp, read_params
from awp_processing.check import check_mesh_cont
from pathlib2 import Path

# !Check these cons in pmcl3d_cons.h in the source code
BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z = 2, 2, 4
nbit_float = 4

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="", help="configuration file")
parser.add_argument("--conf_file", default="param.sh", help="configuration file")
parser.add_argument("--batch_file", default="run.lsf", help="batch script")
args = parser.parse_args()

C = awp.Scenario(model=args.model, conf_file=args.conf_file)
cfg = C.cfg 

# Convert parameters to floats or integers
"""
for k, v in cfg.items():
    if not isinstance(v, Iterable):
        print(k, v, type(v))
        if type(v) == str and v and v.isdigit():
            cfg[k] = float(v) if "." in v else int(v)
    else:
        print(k, v, type(v[0]))
        if not isinstance(v, str) and type(v[0]) == str and v[0].isdigit():
        # is list
            v = [float(x) if "." in v else int(x) for x in v ]
            cfg[k] = v
"""


# output directories
assert Path(args.model, cfg.chkfile).parent.exists() 
assert Path(args.model, cfg.out).exists()

# layers
assert len(cfg.z) == len(cfg.nbgx) == len(cfg.dh) == len(cfg.nsrc) == cfg.g
for i in range(cfg.g):
    assert cfg.x[i] % cfg.px == 0 and cfg.x[i] // cfg.px % BLOCK_SIZE_X == 0, f"Layer-{i}: Mismatch in X"
    assert cfg.y[i] % cfg.py == 0 and cfg.y[i] // cfg.py % BLOCK_SIZE_Y == 0, f"Layer-{i}: Mismatch in Y"
    assert cfg.z[i] // cfg.pz % BLOCK_SIZE_Z == 0, f"Layer-{i}: Mismatch in Z"
    if cfg.insrc != "":
        assert Path(args.model, cfg.insrc + "_" + str(i)).exists(), f"Layer-{i}: Source does not exist"
        assert Path(args.model, cfg.insrc + "_" + str(i)).stat().st_size == cfg.nsrc[i] * (cfg.nst * 6 + 3) * nbit_float, f"Layer-{i}: Mismatch in source size"
    assert Path(args.model, cfg.invel + "_" + str(i)).exists(), f"Layer-{i}: Mesh does not exist"
    assert Path(args.model, cfg.invel + "_" + str(i)).stat().st_size == cfg.x[i] * cfg.y[i] * cfg.z[i] * cfg.nvar * nbit_float, f"Layer-{i}: Mismatch of mesh size"
    if i + 1 < cfg.g:
        # Check consistency of adjcent meshes
        check_mesh_cont(Path(args.model, cfg.invel + "_" + str(i)),
                        Path(args.model, cfg.invel + "_" + str(i + 1)),
                        cfg.x[i], cfg.y[i], cfg.z[i])
    
# Topography
if cfg.intopo:
    file_topo = Path(args.model, cfg.intopo)
    nx, ny, pad = np.fromfile(file_topo, dtype='int32', count=3)
    assert nx == cfg.x[0] and ny == cfg.y[0], f"Mismatch topography domain size" 
    assert (nx + 2 * pad) * (ny + 2 * pad) * nbit_float == file_topo.stat().st_size, f"Topography size does not match parameters"

# Receivers
if cfg.recvfile:
    assert Path(args.model, cfg.recvfile).parent.exists(), f"Receiver output directory does not exist"
    assert cfg.recv_steps % (cfg.recv_stride * cfg.recv_cpu_buffer_size \
            * cfg.recv_gpu_buffer_size * cfg.recv_num_writes) == 0, "Check divisibility of receiver writing"
    assert cfg.recv_length <= len(cfg.recv_coords), f"More receivers required than given"

# Source files in Ossian's format
if cfg.sourcefile:
    assert Path(args.model, cfg.sourcefile).parent.exists(), f"Source file doesn't exist"
    assert cfg.src_steps % (cfg.src_stride * cfg.src_cpu_buffer_size \
            * cfg.src_gpu_buffer_size * cfg.src_num_writes) == 0, f"Check divisibility of source reading"
    assert cfg.src_length == len(cfg.src_coords), f"Mismatch number of sources"
    for suf in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']:
        assert cfg.src_length * cfg.src_steps * nbit_float == Path(args.model, cfg.src_file + "_" + suf).stat().st_size, f"Input source file size doesn't match"




