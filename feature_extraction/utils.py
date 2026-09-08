import numpy as np
from numpy.linalg import norm
import scipy.signal as ss

from utilsKinematics import OpenSimModelWrapper

# from https://stackoverflow.com/a/13849249
def angle_between(v1, v2):
    # gets the angle between two vectors
    v1_u = v1 / norm(v1)
    v2_u = v2 / norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_all(s1, s2):
    # gets the angles between two vectors over time
    assert s1.shape == s2.shape
    out = np.empty(s1.shape[0])
    for i in range(s1.shape[0]):
        out[i] = angle_between(s1[i,:], s2[i,:])
    return out


def trc_arm_angles(xyz, markers, medfilt=11):
    # shoulder, elbow, and wrist markers
    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    re = xyz[:,np.argmax(markers=='r_melbow_study'),:]
    le = xyz[:,np.argmax(markers=='L_melbow_study'),:]
    rw = xyz[:,np.argmax(markers=='r_mwrist_study'),:]
    lw = xyz[:,np.argmax(markers=='L_mwrist_study'),:]

    # gravity vector
    grav = np.zeros_like(rs)
    grav[:,1] = -1

    # shoulder and elbow angles
    rsa = angle_between_all(re-rs, grav) * 180 / np.pi
    rea = angle_between_all(rw-re, re-rs) * 180 / np.pi
    lsa = angle_between_all(le-ls, grav) * 180 / np.pi
    lea = angle_between_all(lw-le, le-ls) * 180 / np.pi

    if medfilt:
        rsa = ss.medfilt(rsa, medfilt)
        rea = ss.medfilt(rea, medfilt)
        lsa = ss.medfilt(lsa, medfilt)
        lea = ss.medfilt(lea, medfilt)

    return rsa, rea, lsa, lea


def center_of_mass(modelPath, motionPath):
    model = OpenSimModelWrapper(modelPath, motionPath)
    df_com = model.get_center_of_mass_values()
    return df_com[['z', 'y', 'x']].values


def center_of_mass_vel(modelPath, motionPath):
    model = OpenSimModelWrapper(modelPath, motionPath)
    df_com = model.get_center_of_mass_speeds()
    return df_com[['z', 'y', 'x']].values


def segment_gait_cycles(xyz, markers, fps):
    ra = xyz[:,np.argmax(markers=='r_ankle_study'),:].copy()
    la = xyz[:,np.argmax(markers=='L_ankle_study'),:].copy()
    ma = (ra - la)
    h = ma[:,1].copy()

    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)
    h = ss.convolve(h, win, mode='same')

    rhjc = xyz[:,np.argmax(markers=='RHJC_study'),:].copy()
    lhjc = xyz[:,np.argmax(markers=='LHJC_study'),:].copy()
    com = (rhjc + lhjc) / 2
    dist = norm(com - com[-1,:], axis=1)
    h[dist > 7] = 0

    h /= h.ptp()

    rlocs, _ = ss.find_peaks(h, height=0.4*h.max(), distance=fps/4)
    llocs, _ = ss.find_peaks(-h, height=-0.4*h.min(), distance=fps/4)

    half_cycles = []
    for rl in rlocs:
        if np.any(llocs > rl):
            ll = llocs[np.argmax(llocs > rl)]
            half_cycles.append((rl, ll))
    for ll in llocs:
        if np.any(rlocs > ll):
            rl = rlocs[np.argmax(rlocs > ll)]
            half_cycles.append((ll, rl))

    # calculate half stride time
    hst = np.array([lb-la for (la, lb) in half_cycles])
    med_hst = np.median(hst)
    pct_errors = np.abs(hst-med_hst)/med_hst
    half_cycles = np.array(half_cycles)[pct_errors < 0.3]

    # recompute median after culling bad cycles
    hst = np.array([lb-la for (la, lb) in half_cycles])
    med_hst = np.median(hst)
    half_cycles = half_cycles[np.argsort(half_cycles[:,0])]

    full_cycles = []
    for k in range(half_cycles.shape[0]-1):
        la, lb = half_cycles[k]
        lc, ld = half_cycles[k+1]
        if lc == lb:
            full_cycles.append((la, ld))
    full_cycles = np.array(full_cycles)

    return half_cycles, full_cycles, h
