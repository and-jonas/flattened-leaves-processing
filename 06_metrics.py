
import importlib

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import utils.base as base_utils
importlib.reload(base_utils)
import os
from scipy import ndimage as ndi


base_path = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")

path_seg_mask = base_path / "predictions/symptoms_seg/20230604_132542_ESWW0070020_1.png"
path_det_mask = base_path / "predictions/symptoms_det/20230604_132542_ESWW0070020_1.png"
path_rgb_img = base_path / "inference_crops/20230604_132542_ESWW0070020_1.JPG"

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
l_mask = np.where(mask == 2, 255, 0).astype("uint8")
l_mask = base_utils.filter_objects_size(mask=l_mask, size_th=500, dir="smaller")

# fill small holes
kernel = np.ones((3, 3), np.uint8)
l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_DILATE, kernel, iterations=2)
l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_ERODE, kernel, iterations=2)
# remove some artifacts, e.g., around insect damage
l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_ERODE, kernel, iterations=1)
l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_DILATE, kernel, iterations=1)
# reformat
l_mask = np.where(l_mask, 255, 0).astype("uint8")

# plt.imshow(l_mask)
# plt.show()

# ==================================================================================================================
# 2. Analyze leaf
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
contours, _ = cv2.findContours(l_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

# ==================================================================================================================
# 3. Analyze lesions
# ==================================================================================================================

# # if not lesions are found, the original image without overlay is saved
# if len(contours) < 1:
#     continue

# # Process each detected object in the current frame
# img = cv2.imread(str(path_rgb_img), cv2.IMREAD_COLOR)
# checker = copy.copy(img)

lesion_data = []

# prepare distance map
mask_invert = np.bitwise_not(l_mask)
distance_invert = ndi.distance_transform_edt(mask_invert)

# # for pycnidiation monitoring
# image_with_pycn = copy.copy(img)

# prepare pycnidia density map
pycn_distmap, pycn_dmap = base_utils.get_pycnidia_maps(
    mask=mask,
    resize_factor=5, bandwidth=25, kernel='gaussian'
)

# plt.imshow(pycn_dmap)
# plt.show()

instance_mask = np.zeros_like(mask)
for idx, contour in enumerate(contours):

    # print("----" + str(idx))

    # get the roi
    # x, y, w, h = map(int, cv2.boundingRect(contour))
    rect = cv2.boundingRect(contour)
    roi = base_utils.select_roi_2(rect=rect, mask=l_mask)
    
    # check if is fully on the imaged leaf
    in_leaf_checker = np.unique(mask_leaf[np.where(roi)[0], np.where(roi)[1]])[0]

    instance_mask = np.where(roi, idx + 1, instance_mask)

    # modify dimensions of the bounding rectangle
    rect = base_utils.get_bounding_boxes(rect=rect)

    # # extract roi
    # _, empty_img, _ = lesion_utils.select_roi(rect=rect, img=img, mask=l_mask)

    # extract lesion data
    # these are extracted from the original (un-smoothed) contour
    contour_area = cv2.contourArea(contour)
    contour_perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_solidity = float(contour_area) / hull_area
    _, _, w, h = x, y, w, h = cv2.boundingRect(contour)

    # extract pycnidia number
    pycn_mask = np.where(roi, mask, 0)
    n_pycn = len(np.where(pycn_mask == 5)[0])
    pycn_density_lesion = n_pycn / contour_area

    # extract rust number
    rust_mask = np.where(roi, mask, 0)
    n_rust = len(np.where(rust_mask == 6)[0])
    rust_density_lesion = n_rust / contour_area

    pycn_features, pycn_contour = base_utils.get_pycn_features(
        mask=mask,
        lesion_mask=roi,
        contour=contour,
        max_dist=50, bandwidth=25, kernel='gaussian'
    )

    # collect output data
    ldat = {'area': contour_area,
            'perimeter': contour_perimeter,
            'solidity': contour_solidity,
            'max_width': w,
            'max_height': h,
            'n_pycn': n_pycn,
            'pycn_density_lesion': pycn_density_lesion,
            'rust_density_lesion': rust_density_lesion
            }

    # draw pycnidiation contour
    if pycn_contour is not None:
        for p in pycn_contour:
            cv2.drawContours(image_with_pycn, p, -1, (255, 255, 0), 2)
    cv2.drawContours(image_with_pycn, contour, -1, (0, 0, 0), 2)

    # data
    ldat = ldat | pycn_features
    lesion_data.append(ldat)

# except:
#     self.log_fail(sample_name)
#     continue

# export instance mask
cv2.imwrite(f'{out_paths[8]}/{png_name}', instance_mask)