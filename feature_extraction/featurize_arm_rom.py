
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from utilsLoaders import read_trc



def reachable_area_right(rs, ls, rw):
    # center on right shoulder
    rw = rw - rs
    ls = ls - rs

    # remove torso yaw and roll
    roty = np.arctan(ls[:,2] / ls[:,0])
    rotz = np.arctan(ls[:,1] / norm(ls[:,[0,2]], axis=1))
    for i in range(rw.shape[0]):
        rot = R.from_euler('yz', [roty[i], rotz[i]])
        rw[i,:] = rot.apply(rw[i,:])

    # clip contralateral crossover
    rw[:,0] = rw[:,0].clip(0, None)

    # compute reachable area
    ch = ConvexHull(rw)
    reachable_area = ch.area

    return reachable_area


def arm_rom_trc_feats(xyz, markers):
    # get shoulder and wrist marker trajectories
    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    rw = xyz[:,np.argmax(markers=='r_mwrist_study'),:]
    lw = xyz[:,np.argmax(markers=='L_mwrist_study'),:]

    # compute right arm reachable reachable_area
    rw_area_r = reachable_area_right(rs, ls, rw)

    # sagittal flip
    ls[:,0] *= -1
    rs[:,0] *= -1
    lw[:,0] *= -1
    rw_area_l = reachable_area_right(ls, rs, lw)

    # sum right and left scores
    rw_area = rw_area_r + rw_area_l

    return {
            'arm_rom_rw_area': float(rw_area),
           }

def feats_arm_rom(trc_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    feats = arm_rom_trc_feats(xyz, markers)
    return feats


if __name__ == '__main__':
    feats = feats_arm_rom(snakemake.input['trc'])
    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


