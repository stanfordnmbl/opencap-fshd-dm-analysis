import numpy as np
import scipy.signal as ss
from numpy.linalg import norm

from utilsLoaders import read_trc, read_mot
from utils import center_of_mass, center_of_mass_vel
from utils import segment_gait_cycles


def gait_trc_feats(xyz, markers, fps, com, comv, activity):
    half_cycles, full_cycles, h = segment_gait_cycles(xyz, markers, fps)

    # compute usable kinematics zone
    com_dist = norm((com - com[-2,:])[:,[0,2]], axis=-1)
    zone_start = np.argmax(com_dist < 4)
    zone_stop = np.argmax(com_dist < 1)
    zone = np.arange(xyz.shape[0])
    zone = (zone >= zone_start) & (zone < zone_stop)

    # LP filter kernel
    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)

    # transform to basis aligned with walk direction
    direc = com[-1,:] - com[0,:]
    direc /= norm(direc)
    pos_z = np.array([0.0, 1.0, 0.0])
    agrav = pos_z - (pos_z @ direc) / (direc @ direc) * direc
    agrav /= norm(agrav)
    perp = np.cross(agrav, direc)
    perp /= norm(perp)
    new_basis = np.stack([perp, agrav, direc])
    P = np.linalg.inv(new_basis)
    xyz = np.einsum('...ij,jk->...ik', xyz, P)

    # compute lateral sway
    com_sway = comv[:,0]
    com_sway = ss.convolve(com_sway, win, mode='same')

    # compute lateral lean
    c7 = xyz[:,np.argmax(markers=='r_C7'),:]
    rh = xyz[:,np.argmax(markers=='RHJC_study'),:].copy()
    lh = xyz[:,np.argmax(markers=='LHJC_study'),:].copy()
    midhip = (rh + lh) / 2
    trunk = c7 - midhip
    trunk_tilt = np.arctan2(trunk[:,0], trunk[:,1]) * 180/np.pi
    trunk_tilt = ss.convolve(trunk_tilt, win, mode='same')

    # compute metrics
    time_3m = (zone_stop-zone_start)/fps
    speed = 3 / time_3m
    com_sway = com_sway[zone].std()
    trunk_lean = np.mean(np.abs(trunk_tilt[zone]))

    stride_time = np.diff(full_cycles, 1).mean() / fps

    ra = xyz[:,np.argmax(markers=='r_ankle_study'),:].copy()
    la = xyz[:,np.argmax(markers=='L_ankle_study'),:].copy()

    stride_lens = []
    ankle_elevs = []
    for cyc in full_cycles:
        if h[cyc[0]] > 0:
            lenny = norm(np.diff(la[cyc],0))
        else:
            lenny = norm(np.diff(ra[cyc],0))
        stride_lens.append(lenny)

        # find ankle elevation at mid-swing
        assert len(cyc) == 2
        ia, ib = cyc[0], cyc[1]
        ms = ia + np.argmin(np.abs(la[ia:ib,2]-ra[ia:ib,2]))
        ankle_elevs.append(np.abs(la[ms]-ra[ms]))
    stride_len = np.median(stride_lens)
    ankle_elev = np.median(ankle_elevs)

    return {
            f'{activity}_speed': float(speed),
            f'{activity}_com_sway': float(com_sway),
            f'{activity}_stride_time': float(stride_time),
            f'{activity}_stride_len': float(stride_len),
            f'{activity}_trunk_lean': float(trunk_lean),
            f'{activity}_ankle_elev': float(ankle_elev),
           }


def gait_mot_feats(df, activity):

    rha = df.hip_adduction_r.to_numpy()
    lha = df.hip_adduction_l.to_numpy()
    rka = df.knee_angle_r.to_numpy()
    lka = df.knee_angle_l.to_numpy()

    rha = ss.medfilt(rha, 11)
    lha = ss.medfilt(lha, 11)
    rka = ss.medfilt(rka, 11)
    lka = ss.medfilt(lka, 11)

    ptp_r_hip_add = rha.ptp()
    ptp_l_hip_add = lha.ptp()
    mean_ptp_hip_add = (ptp_r_hip_add + ptp_l_hip_add) / 2

    max_rka = rka.max()
    max_lka = lka.max()
    mean_max_ka = (max_rka + max_lka) / 2

    return {
            f'{activity}_mean_ptp_hip_add': float(mean_ptp_hip_add),
            f'{activity}_mean_max_ka': float(mean_max_ka),
           }


def featurize_gait(trc_fpath, mot_fpath, model_fpath, activity='gait'):
    fps, markers, xyz = read_trc(trc_fpath)

    com = center_of_mass(model_fpath, mot_fpath)
    comv = center_of_mass_vel(model_fpath, mot_fpath)

    trc_feats = gait_trc_feats(xyz, markers, fps, com, comv, activity)

    df = read_mot(mot_fpath)
    mot_feats = gait_mot_feats(df, activity)

    feats = trc_feats.copy()
    feats.update(mot_feats)
    return feats

