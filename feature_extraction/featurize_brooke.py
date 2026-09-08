from pathlib import Path

import numpy as np
import pandas as pd

from utilsLoaders import read_trc
from utils import trc_arm_angles


def brooke_trc_feats(xyz, markers):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)
    mean_sa = (rsa + lsa) / 2
    mean_ea = (rea + lea) / 2
    max_mean_sa = np.max(mean_sa)

    min_sa = np.vstack([rsa, lsa]).min(0)
    max_min_sa = min_sa.max()

    max_ea = np.vstack([rea, lea]).max(0)
    max_ea_at_max_min_sa = max_ea[np.argmax(min_sa)]

    max_sa_ea_ratio = np.max(mean_sa / (mean_ea+90))
    return {
            'brooke_max_mean_sa': float(max_mean_sa),
            'brooke_max_min_sa': float(max_min_sa),
            'brooke_max_ea_at_max_min_sa': float(max_ea_at_max_min_sa),
            'brooke_max_sa_ea_ratio': float(max_sa_ea_ratio),
           }


def feats_brooke(trc_fpath, sto_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    feats = brooke_trc_feats(xyz, markers)
    return feats


if __name__ == '__main__':
    feats = feats_brooke(snakemake.input['trc'], None)

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)

