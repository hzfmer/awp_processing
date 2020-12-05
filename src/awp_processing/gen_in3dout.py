import argparse
from pathlib2 import Path
from awp_processing import read_params


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="", help="Scenario directory")
parser.add_argument("--conf_file", default="param.sh", help="Configuration file")


args = parser.parse_args()
#C = awp.Scenario(model=args.model, conf_file=args.conf_file)

cfg = read_params.read_params(Path(args.model, args.conf_file)) 

with open(Path(args.model, "IN3D.out"), "w") as fid:
    fid.write(f"{cfg.nbgx[0]} NBGX\n")
    fid.write(f"{cfg.nbgy[0]} NBGY\n")
    fid.write(f"{cfg.nbgz[0]} NBGZ\n")
    fid.write(f"{cfg.nedx[0]} NEDX\n")
    fid.write(f"{cfg.nedy[0]} NEDY\n")
    fid.write(f"{cfg.nedz[0]} NEDZ\n")
    fid.write(f"{cfg.nskpx[0]} NSKPX\n")
    fid.write(f"{cfg.nskpy[0]} NSKPY\n")
    fid.write(f"{cfg.nskpz[0]} NSKPZ\n")
    fid.write(f"{cfg.dt} DT\n")
    fid.write(f"{cfg.tmax + 0.00001} TMAX\n")
    fid.write(f"{cfg.tskip} NTISKP\n")
    fid.write(f"{cfg.sxrgo} SXRGO\n")
    fid.write(f"{cfg.syrgo} SYRGO\n")
    fid.write(f"{cfg.szrgo} SZRGO\n")
    fid.write(f"{cfg.read_step} READ_STEP\n")
    fid.write(f"{cfg.wstep} WRITE_STEP\n")
    fid.write(f"{cfg.ivelocity} IVELOCITY\n")
