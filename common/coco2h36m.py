"""Map a 2D detector's COCO-17 keypoints to the repo's H36M-17 convention.

RTMPose / most 2D detectors output COCO-17:
  0 nose 1 Leye 2 Reye 3 Lear 4 Rear 5 Lsho 6 Rsho 7 Lelb 8 Relb 9 Lwri 10 Rwri
  11 Lhip 12 Rhip 13 Lkne 14 Rkne 15 Lank 16 Rank

H36M-17 (this repo): 0 Hip 1 RHip 2 RKnee 3 RAnkle 4 LHip 5 LKnee 6 LAnkle
  7 Spine 8 Thorax 9 Neck 10 Head 11 LSho 12 LElb 13 LWri 14 RSho 15 RElb 16 RWri

Hip/Spine/Thorax/Neck/Head are synthesized (MotionBERT `coco2h36m` convention),
since COCO has no spine/thorax. Confidence for a synthesized joint is the min of
its sources (conservative → low conf if any contributor is occluded).
"""

import numpy as np


def coco2h36m_kpts(x):
    """x: (T,17,2) COCO → (T,17,2) H36M."""
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 11] + x[:, 12]) * 0.5      # Hip = mid hips
    y[:, 1] = x[:, 12]                          # RHip
    y[:, 2] = x[:, 14]                          # RKnee
    y[:, 3] = x[:, 16]                          # RAnkle
    y[:, 4] = x[:, 11]                          # LHip
    y[:, 5] = x[:, 13]                          # LKnee
    y[:, 6] = x[:, 15]                          # LAnkle
    y[:, 8] = (x[:, 5] + x[:, 6]) * 0.5         # Thorax = mid shoulders
    y[:, 7] = (y[:, 0] + y[:, 8]) * 0.5         # Spine = mid(hip, thorax)
    y[:, 9] = x[:, 0]                           # Neck ~ nose
    y[:, 10] = (x[:, 1] + x[:, 2]) * 0.5        # Head = mid eyes
    y[:, 11] = x[:, 5]                          # LShoulder
    y[:, 12] = x[:, 7]                          # LElbow
    y[:, 13] = x[:, 9]                          # LWrist
    y[:, 14] = x[:, 6]                          # RShoulder
    y[:, 15] = x[:, 8]                          # RElbow
    y[:, 16] = x[:, 10]                         # RWrist
    return y


def coco2h36m_conf(c):
    """c: (T,17) COCO scores → (T,17) H36M (min over contributors for synth joints)."""
    y = np.zeros_like(c)
    y[:, 0] = np.minimum(c[:, 11], c[:, 12])
    y[:, 1] = c[:, 12]; y[:, 2] = c[:, 14]; y[:, 3] = c[:, 16]
    y[:, 4] = c[:, 11]; y[:, 5] = c[:, 13]; y[:, 6] = c[:, 15]
    y[:, 8] = np.minimum(c[:, 5], c[:, 6])
    y[:, 7] = np.minimum(y[:, 0], y[:, 8])
    y[:, 9] = c[:, 0]
    y[:, 10] = np.minimum(c[:, 1], c[:, 2])
    y[:, 11] = c[:, 5]; y[:, 12] = c[:, 7]; y[:, 13] = c[:, 9]
    y[:, 14] = c[:, 6]; y[:, 15] = c[:, 8]; y[:, 16] = c[:, 10]
    return y


def normalize_screen(kp, w, h):
    """Pixel → [-1,1] (VideoPose3D convention, matches training)."""
    return kp / w * 2 - np.array([1.0, h / w], dtype=np.float32)
