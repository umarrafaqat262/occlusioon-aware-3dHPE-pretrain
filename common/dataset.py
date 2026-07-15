"""H36M PyTorch Dataset.

Reads the standard H36M pickle file (same format used by MotionBERT /
PoseMamba): a dict with 'train'/'test' keys, each containing:
  joint_2d       (N, 17, 2)  pixel coords
  joint3d_image  (N, 17, 3)  3D coords in image space (mm)
  confidence     (N, 17) or (N, 17, 1)  CPN/SH joint confidence
  camera_name    (N,)  one of '54138969','60457274','55011271','58860488'
  source         (N,)  video clip id strings  (optional, for slicing)

Camera resolutions:
  '54138969' / '60457274': 1000 × 1002
  '55011271' / '58860488': 1000 × 1000
"""

import os, pickle, random
import numpy as np
import torch
from torch.utils.data import Dataset


_CAM_RES = {
    '54138969': (1000, 1002),
    '60457274': (1000, 1002),
    '55011271': (1000, 1000),
    '58860488': (1000, 1000),
}


def _split_clips(vid_list, n_frames, stride):
    """Return index array of shape (n_clips, n_frames) with given stride."""
    indices = []
    db = {}
    for i, vid in enumerate(vid_list):
        db.setdefault(vid, []).append(i)
    for vid, frames in db.items():
        frames = np.array(frames)
        for start in range(0, len(frames) - n_frames + 1, stride):
            indices.append(frames[start:start + n_frames])
    return np.array(indices, dtype=np.int64)


class H36MDataset(Dataset):
    def __init__(self, data_dir, split, num_frames, stride, dt_file):
        path = os.path.join(data_dir, dt_file)
        with open(path, 'rb') as f:
            dt = pickle.load(f)

        raw      = dt[split]
        joint_2d = raw['joint_2d'].astype(np.float32)          # (N, 17, 2)
        joint_3d = raw['joint3d_image'].astype(np.float32)      # (N, 17, 3)
        cam_names = raw['camera_name']                          # (N,)

        # Confidence: (N, 17) or (N, 17, 1)
        if 'confidence' in raw:
            conf = raw['confidence'].astype(np.float32)
            if conf.ndim == 2:
                conf = conf[:, :, None]                         # (N, 17, 1)
        else:
            conf = np.ones((len(joint_2d), 17, 1), dtype=np.float32)

        # Normalise 2D coords to [-1, 1] per camera
        for i, cam in enumerate(cam_names):
            res_w, res_h = _CAM_RES.get(cam, (1000, 1000))
            joint_2d[i, :, 0] = joint_2d[i, :, 0] / res_w * 2 - 1
            joint_2d[i, :, 1] = joint_2d[i, :, 1] / res_w * 2 - res_h / res_w

        # Normalise 3D coords to [-1, 1] scale
        for i, cam in enumerate(cam_names):
            res_w, _ = _CAM_RES.get(cam, (1000, 1000))
            joint_3d[i, :, :2] = joint_3d[i, :, :2] / res_w * 2 - 1
            joint_3d[i, :,  2] = joint_3d[i, :,  2] / res_w * 2

        # Root-centre 3D
        joint_3d = joint_3d - joint_3d[:, :1, :]               # (N, 17, 3)

        # Slice into clips
        source = raw.get('source', np.arange(len(joint_2d)).astype(str))
        clips  = _split_clips(source, num_frames, stride)       # (n_clips, T)

        self.pose_2d = joint_2d[clips]   # (n_clips, T, 17, 2)
        self.pose_3d = joint_3d[clips]   # (n_clips, T, 17, 3)
        self.conf    = conf[clips]        # (n_clips, T, 17, 1)

    def __len__(self):
        return len(self.pose_2d)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.pose_2d[idx]),
            torch.from_numpy(self.pose_3d[idx]),
            torch.from_numpy(self.conf[idx]),
        )
