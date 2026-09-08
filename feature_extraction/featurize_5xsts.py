
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

from utilsLoaders import read_trc
from utils import angle_between_all


def sts_trc_feats(xyz, markers, fps):
    c7 = xyz[:,np.argmax(markers=='C7_study'),:]
    rh = xyz[:,np.argmax(markers=='RHJC_study'),:]
    lh = xyz[:,np.argmax(markers=='lHJC_study'),:]
    mh = (rh + lh) / 2

    h = mh[:,1].copy()
    h -= h.min()
    h /= h.max()
    locs, _ = ss.find_peaks(h, height=0.75, prominence=0.5)

    la = locs[0] - np.argmax(h[locs[0]::-1] < 0.25)
    if la == locs[0]:
        la = np.argmin(h[:locs[0]])
    lb = locs[-1] + np.argmax(h[locs[-1]:] < 0.25)
    if lb == locs[-1]:
        lb = locs[-1] + np.argmin(h[locs[-1]:])

    if len(locs) > 1:
        tdiffs = np.diff(locs) / fps
        sts_time = np.median(tdiffs)
    else:
        sts_time = (lb - la)/fps

    sts_time_5 = (lb - la)/fps / len(locs) * 5

    # gravity vector
    grav = np.zeros_like(c7)
    grav[:,1] = -1

    trunk_angle = angle_between_all(mh-c7, grav) * 180 / np.pi

    # smooth trunk angle with 0.5s hann window
    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)
    trunk_angle = ss.convolve(trunk_angle, win, mode='same')

    lean_maxs = []
    if len(locs) > 1:
        for i in range(len(locs)-1):
            la, lb = locs[i], locs[i+1]
            seg = trunk_angle[la:lb]
            lean_maxs.append(seg.max())
    else:
        seg = trunk_angle[la:lb]
        lean_maxs.append(seg.max())

    lean_max = np.mean(lean_maxs)

    # measure width of base of support
    rank = xyz[:,np.argmax(markers=='r_ankle_study'),:]
    lank = xyz[:,np.argmax(markers=='L_ankle_study'),:]
    ankle_dist = np.linalg.norm(lank - rank, axis=1)
    stance_width = ankle_dist[la:lb].ptp()

    return {
            '5xsts_time_5': float(sts_time_5),
            '5xsts_lean_max': float(lean_max),
            '5xsts_stance_width': float(stance_width),
           }


def feats_5xsts(trc_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    feats = sts_trc_feats(xyz, markers, fps)
    return feats


if __name__ == '__main__':
    feats = feats_5xsts(snakemake.input['trc'])

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)

