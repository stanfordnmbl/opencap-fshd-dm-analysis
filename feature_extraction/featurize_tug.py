from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss
from numpy.linalg import norm

from utilsLoaders import read_trc


def tug_trc_feats(xyz, markers, fps):
    rh = xyz[:,np.argmax(markers=='RHJC_study'),:]
    lh = xyz[:,np.argmax(markers=='LHJC_study'),:]
    mh = (rh + lh) / 2
    h = mh[:,1]
    h -= h.min()
    h /= h.max()

    if not np.any(h[:int(fps)] < 0.3):
        la = np.nan
    else:
        la = np.argmax(h > 0.3)
    if not np.any(h[-int(fps):] < 0.3):
        lb = np.nan
    else:
        lb = len(h) - np.argmax(h[-1::-1] > 0.7)
        lb = lb + np.argmax(h[lb:] < 0.3)

    tug_time = (lb - la)/fps

    # compute angular velocity from shoulder heading
    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    rsls = rs-ls
    heading = np.arctan2(rsls[:,0], rsls[:,2]) * 180 / np.pi
    heading = np.unwrap(heading)
    avel = np.diff(heading, prepend=heading[0]) * fps

    # filter angular velocity with hann kernel
    W = int(1*fps)
    kernel = ss.windows.hann(W)
    kernel /= np.sum(kernel)
    avel = ss.convolve(avel, kernel, mode='same')

    ms = (rs + ls)/2
    dist = norm(ms - ms[0], axis=1)
    win = (dist.max() - dist) < 1

    turn_avel = np.abs(np.mean(avel[win]))
    turn_max_avel = np.max(np.abs(avel[win]))

    return {
            'tug_cone_time': float(tug_time),
            'tug_cone_turn_avel': float(turn_avel),
            'tug_cone_turn_max_avel': float(turn_max_avel),
           }


def feats_tug(trc_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    feats = tug_trc_feats(xyz, markers, fps)
    return feats


if __name__ == '__main__':
    feats = feats_tug(snakemake.input['trc'])

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)

