
import importlib

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import utils.base as base_utils
importlib.reload(base_utils)

base_path = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")

path_seg_mask = base_path / "predictions/symptoms_seg/20230604_132542_ESWW0070020_1.png"
path_det_mask = base_path / "predictions/symptoms_det/20230604_132542_ESWW0070020_1.png"

out_path = base_path / "metrics"

# ==================================================================================================================
# 1. Pre-processing
# ==================================================================================================================

# get names
base_name = os.path.basename(path_seg_mask)
jpg_name = base_name.replace(".png", ".JPG")
stem_name = base_name.replace(".png", "")
leaf_name = "_".join(stem_name.split("_")[2:4])
csv_name = base_name.replace(".png", ".csv")

# get masks
seg_mask = cv2.imread(str(path_seg_mask), cv2.IMREAD_UNCHANGED)
det_mask = cv2.imread(str(path_det_mask), cv2.IMREAD_UNCHANGED)

# LEAF_BACKGROUND = 0
# LEAF = 1
# LESION = 2
# INSECT_DAMAGE = 3
# POWDERY_MILDEW = 4
# PYCNIDIA = 5
# RUST = 6

# pycnidia and rust pustules are denoted as 1 and 2
# to combine, change to 5 and 6
det_mask = np.where(det_mask == 0, det_mask, det_mask + 4)

# combine masks
mask = np.where(det_mask == 0, seg_mask, det_mask)

# get leaf mask (without insect damage!)
mask_leaf = np.where((mask >= 1) & (mask != 3), 1, 0).astype("uint8")

# get lesion mask
frame = np.where(mask == 2, 255, 0).astype("uint8")
frame = base_utils.filter_objects_size(mask=frame, size_th=500, dir="smaller")

# fill small holes
kernel = np.ones((3, 3), np.uint8)
frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=2)
frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=2)
# remove some artifacts, e.g., around insect damage
frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=1)
frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=1)
# reformat
frame = np.where(frame, 255, 0).astype("uint8")

# plt.imshow(frame)
# plt.show()

# ==================================================================================================================
# 5. Analyze leaf
# ==================================================================================================================

# summary stats
la_tot = len(np.where(mask != 0)[0])
la_damaged = len(np.where((mask != 0) & (mask != 1))[0])
la_healthy = len(np.where(mask == 1)[0])
la_damaged_f = la_damaged / la_tot
la_healthy_f = la_healthy / la_tot
la_insect = len(np.where(mask == 3)[0])
la_insect_f = la_insect / la_tot
n_pycn = len(np.where(mask == 5)[0])
rust_idx = np.where(mask == 6)
rust_point_list = base_utils.filter_points(x=rust_idx[1], y=rust_idx[0], min_distance=7)
n_rust = len(rust_point_list)
contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
n_lesion = len(contours)
placl = len(np.where((mask == 2) | (mask == 5))[0]) / (la_tot - la_insect)
pycn_density = n_pycn / (la_tot - la_insect)
rust_density = n_rust / (la_tot - la_insect)

# grab data
leaf_data = [
    {
        'la_tot': la_tot,
        'la_damaged': la_damaged,
        'la_healthy': la_healthy,
        'la_damaged_f': la_damaged_f,
        'la_healthy_f': la_healthy_f,
        'la_insect': la_insect,
        'la_insect_f': la_insect_f,
        'n_pycn': n_pycn,
        'n_rust': n_rust,
        'n_lesion': n_lesion,
        'placl': placl,
        'pycn_density': pycn_density,
        'rust_density': rust_density
    },
]

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(leaf_data)

# Export the DataFrame to a CSV file
df.to_csv(out_path / csv_name, index=False)