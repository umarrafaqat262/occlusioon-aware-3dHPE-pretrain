"""H36M 17-joint skeleton constants."""

# Joint order: Hip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle,
#              Spine, Thorax, Neck, Head,
#              LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# 16 bones: child and parent joint indices
BONE_CHILD_IDX  = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]
BONE_PARENT_IDX = [0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15]

# Symmetric left-right pairs BY BONE INDEX (bone i has child joint i+1).
# R-leg bones 0,1,2 ↔ L-leg bones 3,4,5 ; L-arm bones 10,11,12 ↔ R-arm bones 13,14,15.
BONE_SYMMETRY_PAIRS = [(0, 3), (1, 4), (2, 5), (10, 13), (11, 14), (12, 15)]

# Kinematic-tree scan order (novelty A): a root→leaf DFS pre-order of the H36M
# skeleton. Because the joints were indexed in DFS order (parent index < child
# index for every joint), the natural index order IS the kinematic tree order:
# the forward spatial SSM scan therefore propagates state root→leaf along bones,
# and the reversed scan propagates leaf→root. Kept explicit (not hard-coded as
# range(17)) so ablations can swap in a non-anatomical order.
KIN_SCAN_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# inverse permutation to restore canonical joint order after a reordered scan
KIN_SCAN_INV = [KIN_SCAN_ORDER.index(j) for j in range(len(KIN_SCAN_ORDER))]

# Undirected skeleton edges (child, parent) — used to build the graph adjacency
# for the KPA graph conv, the SSI state-fusion adjacency, and the Laplacian PE.
SKELETON_EDGES = [(j, p) for j, p in enumerate(H36M_PARENTS) if p >= 0]

# Limb-chain gather index (PoseMamba global-local "reordering", verified from the
# official code). Length 17; the root (0) and thorax (8) repeat as branch points.
# A limb-ordered VIEW of the joints is produced by x[:, LIMB_REORDER_INDEX] and
# fused additively with the natural (global) order before the spatial SSM scan, so
# the recurrence traverses anatomically adjacent joints along each limb chain:
#   right leg 0->1->2->3 | left leg 0->4->5->6 | left arm 8->11->12->13 | right arm 8->14->15->16
LIMB_REORDER_INDEX = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]

