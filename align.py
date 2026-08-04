
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from natsort import natsorted
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

def _uid_from_path(p: str) -> str:
    parts = os.path.splitext(os.path.basename(p))[0].split("_")
    return "_".join(parts[2:4])

def get_series(path_images: str, leaf_uid: Optional[str] = None) -> List[List[str]]:
    """Collect sequences of images grouped by sample id.

    The function glob-matches common image extensions in ``path_images`` and
    groups filenames by the token composed from filename parts 3 and 4
    separated by underscore.

    Args:
        path_images: Directory containing images.
        leaf_uid: If provided, filter to a single sample id.

    Returns:
        A list where each item is a sorted list of filepaths for one sample.
    """
    exts = [".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"]
    images: List[str] = []
    for ext in exts:
        images.extend(glob.glob(f"{path_images}/*{ext}"))
    images = natsorted(images)

    # token used to group images into a time series
    image_image_id = [_uid_from_path(l) for l in images]
    uniques = natsorted(np.unique(image_image_id))

    if leaf_uid is not None:
        uniques = [u for u in uniques if str(u) == str(leaf_uid)]

    id_series: List[List[str]] = []
    for unique_sample in uniques:
        image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
        sample_image_names = [images[i] for i in image_idx]
        # sort to ensure sequential processing of subsequent images
        sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
        id_series.append(sample_image_names)

    return id_series

def read_image(path: str, mode: str = "rgb", downscale: bool = False) -> np.ndarray:
    """Read an image from disk and convert to the requested mode.

    Parameters
    - path: file path
    - mode: "rgb" or "mask"
    """
    if mode == "rgb":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mode == "mask":
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        return img

    raise ValueError(f"Unknown mode: {mode}")


def warp_to_reference(src, M, ref_shape, is_mask=False):
    """Warp `src` (image or mask) into the coordinate system defined by `ref_shape` using affine `M`.

    - `src` : source image or mask to warp (numpy array)
    - `M` : 2x3 affine transform mapping src -> reference coords
    - `ref_shape` : shape (H,W,...) of the reference image
    - `is_mask` : if True, use nearest interpolation to preserve labels
    """
    if M is None:
        return None

    h_ref, w_ref = ref_shape[:2]

    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    # ensure src is in a numeric type acceptable to cv2.warpAffine
    src_dtype = src.dtype
    warped = cv2.warpAffine(src, M, (w_ref, h_ref), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if warped.dtype != src_dtype:
        warped = warped.astype(src_dtype)
    return warped

def get_leaf_mask(mask):
    """
    Get a binary mask of the leaf from the segmentation mask.
    Assumes that the leaf is labeled with 1 in the mask.
    """
    return (mask > 0).astype(np.uint8)

def get_stitch_geometry(img1, img2, M):
    """Compute stitch canvas geometry for two images and affine transform.

    Returns a shifted transform mapping image2 into the canvas, the canvas
    size (width, height) and the top-left coordinates where image1 should be
    placed.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.transform(corners2, M)

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)

    all_corners = np.concatenate([corners1, warped_corners2], axis=0)
    xmin, ymin = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    T = np.array([[1, 0, -xmin], [0, 1, -ymin]], dtype=np.float32)

    M_shifted = T @ np.vstack([M, [0, 0, 1]])
    M_shifted = M_shifted[:2]

    canvas_size = (xmax - xmin, ymax - ymin)
    x0 = -xmin
    y0 = -ymin
    return M_shifted, canvas_size, x0, y0

def stitch_image(img1, img2, M_shifted, canvas_size, x0, y0, x_seam=None, is_mask=False):
    """
    Stitch two images using a predefined affine transform.

    is_mask=True:
        - nearest interpolation
        - preserve class labels

    is_mask=False:
        - linear interpolation
    """

    h1, w1 = img1.shape[:2]
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    warped_img2 = cv2.warpAffine(
        img2,
        M_shifted,
        canvas_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid2 = cv2.warpAffine(
        np.ones(img2.shape[:2], dtype=np.uint8),
        M_shifted,
        canvas_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    canvas1 = np.zeros_like(warped_img2)
    canvas1[y0 : y0 + h1, x0 : x0 + w1] = img1

    valid1 = np.zeros(canvas1.shape[:2], dtype=bool)
    valid1[y0 : y0 + h1, x0 : x0 + w1] = True

    center1 = np.where(valid1)[1].mean()
    center2 = np.where(valid2)[1].mean()

    if x_seam is None:
        overlap = valid1 & valid2
        _, xs = np.where(overlap)
        x_seam = (xs.min() + xs.max()) // 2

    yy, xx = np.indices(valid2.shape)

    if center1 < center2:
        # image 1 is left, image 2 is right
        use_img2 = valid2 & (xx >= x_seam)
    else:
        # image 2 is left, image 1 is right
        use_img2 = valid2 & (xx <= x_seam)

    result = canvas1.copy()
    result[use_img2] = warped_img2[use_img2]

    return result, x_seam


def upscale_affine(M: np.ndarray, scale: float) -> np.ndarray:
    """Undo a coordinate scale applied during feature detection.

    The original script detected features on half-sized images and then
    adjusted the translation components. This helper centralises that logic.
    """
    M_full = M.copy()
    M_full[0, 2] /= scale
    M_full[1, 2] /= scale
    return M_full


# directories
dir_img = "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/renamed/inference_crops"
dir_msk = "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/renamed/predictions/symptoms_seg/pred"

# output directories for aligned results
out_images_dir = "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/renamed/aligned/images"
out_masks_dir = "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/renamed/aligned/masks"

# collect all series for images and masks
all_series_img = get_series(path_images=os.path.join(dir_img))
all_series_msk = get_series(path_images=os.path.join(dir_msk))

# # TEMPORARY: keep only series that contain a file matching the pattern "CH13063920260603.png" (case-insensitive)
# PATTERN = "CH13063920260603"
# all_series_img = [s for s in all_series_img if any(PATTERN in os.path.basename(p) for p in s)]
# all_series_msk = [s for s in all_series_msk if any(PATTERN in os.path.basename(p) for p in s)]

# build mask lookup by uid
mask_lookup = { _uid_from_path(s[0]): s for s in all_series_msk if s }

for series_img in all_series_img:
    if not series_img:
        continue
    uid = _uid_from_path(series_img[0])
    series_msk = mask_lookup.get(uid)
    if not series_msk:
        # no corresponding masks for this uid
        continue

    # load images and masks for this uid
    img = [read_image(p, mode="rgb") for p in series_img]
    msk = [read_image(p, mode="mask") for p in series_msk]
    leaf_masks = [get_leaf_mask(m) for m in msk]

    # -----------------------------------
    # Compute SIFT features
    # -----------------------------------
    sift = cv2.SIFT_create(nfeatures=1500)

    features: List[Dict[str, Any]] = []
    for image, leaf_mask in tqdm(zip(img, leaf_masks), total=min(len(img), len(leaf_masks)), desc=f"SIFT {uid}"):
        image_small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        mask_small = cv2.resize(leaf_mask, (0, 0), fx=0.5, fy=0.5)
        kp, des = sift.detectAndCompute(image_small, (mask_small.astype(np.uint8) * 255))
        features.append({"kp": kp, "des": des})

    # -----------------------------------
    # Matcher
    # -----------------------------------
    bf = cv2.BFMatcher(cv2.NORM_L2)
    results: List[Dict[str, Any]] = []

    for i in range(len(features) - 1):
        des1 = features[i]["des"]
        des2 = features[i + 1]["des"]
        if des1 is None or des2 is None:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        # Lowe ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            continue
        pts1 = np.float32([features[i]["kp"][m.queryIdx].pt for m in good])
        pts2 = np.float32([features[i + 1]["kp"][m.trainIdx].pt for m in good])
        M, inliers = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3)
        if M is None:
            continue
        M_full = upscale_affine(M, 0.5)
        n_inliers = int(inliers.sum())
        inlier_ratio = n_inliers / len(good)
        results.append({
            "pair": (i, i + 1),
            "good_matches": len(good),
            "inliers": n_inliers,
            "ratio": inlier_ratio,
            "transform": M_full,
        })

    # -----------------------------
    # Warp & merge masks for result pairs
    # -----------------------------
    merged_masks = []
    warped_images = []

    for r in results:
        if r["inliers"] < 30:
            continue
        M = r.get("transform")
        i, j = r["pair"]
        M_shifted, canvas_size, x0, y0 = get_stitch_geometry(img[i], img[j], M)
        img_stitched, x_seam = stitch_image(img[i], img[j], M_shifted, canvas_size, x0, y0, is_mask=False)
        mask_stitched, x_seam = stitch_image(msk[i], msk[j], M_shifted, canvas_size, x0, y0, is_mask=True)

        # ensure output directories exist
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)

        # save stitched image (convert RGB->BGR for OpenCV) and mask
        base1 = os.path.splitext(os.path.basename(series_img[i]))[0]
        base2 = os.path.splitext(os.path.basename(series_img[j]))[0]
        fname_img = f"{base1}-{base2}.png"
        fname_mask = f"{base1}-{base2}_mask.png"
        img_out_path = os.path.join(out_images_dir, fname_img)
        mask_out_path = os.path.join(out_masks_dir, fname_mask)

        try:
            cv2.imwrite(img_out_path, cv2.cvtColor(img_stitched, cv2.COLOR_RGB2BGR))
        except Exception:
            cv2.imwrite(img_out_path, img_stitched)

        cv2.imwrite(mask_out_path, mask_stitched)
    
