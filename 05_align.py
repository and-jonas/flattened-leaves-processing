
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from natsort import natsorted
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Manager, Process

from pathlib import Path
import sys

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
    exts = [".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]
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

def get_leaf_mask(mask, min_area=120000):
    """
    Get a cleaned binary leaf mask from a multiclass segmentation mask.
    """

    leaf = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf,
        connectivity=8
    )

    cleaned = np.zeros_like(leaf)

    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned

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
    """Undo a coordinate scale applied during feature detection."""
    M_full = M.copy()
    M_full[0, 2] /= scale
    M_full[1, 2] /= scale
    return M_full


def plot_inlier_matches(img1, img2, kp1, kp2, matches, inliers_mask, scale=0.5, savepath: Optional[str] = None):
    """Plot matched keypoints and highlight inliers.

    - `img1`, `img2` are full-size RGB images as returned by `read_image`.
    - `kp1`, `kp2` are keypoint lists detected on scaled images (scale factor `scale`).
    - `matches` is a list of `cv2.DMatch` between the scaled-image keypoints.
    - `inliers_mask` is an array-like mask (len == len(matches)) marking inliers (truthy).
    - `savepath` optional path to save the figure; when None the plot is shown.
    """
    # create small (detection-size) copies to match keypoint coordinates
    img1_small = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
    img2_small = cv2.resize(img2, (0, 0), fx=scale, fy=scale)

    # Build a mask consumable by cv2.drawMatches: list of 0/1
    mask = None
    try:
        mask = [int(bool(m)) for m in np.asarray(inliers_mask).ravel().tolist()]
    except Exception:
        mask = None

    # drawMatches expects BGR order but we keep RGB everywhere and display with matplotlib
    try:
        drawn = cv2.drawMatches(
            img1_small,
            kp1,
            img2_small,
            kp2,
            matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=mask,
        )
    except Exception:
        # fallback: draw all matches if mask fails
        drawn = cv2.drawMatches(img1_small, kp1, img2_small, kp2, matches, None)

    plt.figure(figsize=(12, 6))
    plt.imshow(drawn)
    plt.axis("off")
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def process_series(series_img, series_msk_seg, series_msk_det, out_aligned_dir, out_matches_dir, out_masks_dir):
    """Process a single leaf series: detect features, match, stitch and save outputs.

    `series_img`, `series_msk_seg`, and `series_msk_det` are lists of file paths (strings).
    """
    if not series_img:
        return {"uid": None, "status": "empty"}

    uid = _uid_from_path(series_img[0])
    if not series_msk_seg and not series_msk_det:
        return {"uid": uid, "status": "no_masks"}

    # load images and masks for this uid
    img = [read_image(p, mode="rgb") for p in series_img]
    msk_seg = [read_image(p, mode="mask") for p in series_msk_seg]
    msk_det = [read_image(p, mode="mask") for p in series_msk_det]
    # get leaf masks
    leaf_masks = [get_leaf_mask(m) for m in msk_seg]
    # get combined masks
    msk = [np.where(det > 0, det + 4, seg) for seg, det in zip(msk_seg, msk_det)]

    # Compute SIFT features
    sift = cv2.SIFT_create(nfeatures=1500)
    features: List[Dict[str, Any]] = []
    for image, leaf_mask in zip(img, leaf_masks):
        image_small = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        mask_small = cv2.resize(leaf_mask, (0, 0), fx=0.5, fy=0.5)
        kp, des = sift.detectAndCompute(image_small, (mask_small.astype(np.uint8) * 255))
        features.append({"kp": kp, "des": des})

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    results: List[Dict[str, Any]] = []
    for i in range(len(features) - 1):
        des1 = features[i]["des"]
        des2 = features[i + 1]["des"]
        if des1 is None or des2 is None:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        # Lowe ratio test
        good = [m for m, n in matches if m.distance < 0.70 * n.distance]
        if len(good) < 35:
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
            "good": good,
            "inliers_mask": inliers.ravel().astype(np.uint8),
        })

    # Warp & merge masks for result pairs
    saved = 0
    for r in results:
        if r["inliers"] < 30:
            continue
        M = r.get("transform")
        i, j = r["pair"]

        # visualize and save match/inliers image (scaled to detector size)
        try:
            os.makedirs(out_matches_dir, exist_ok=True)
            base1 = os.path.splitext(os.path.basename(series_img[i]))[0]
            base2 = os.path.splitext(os.path.basename(series_img[j]))[0]
            matches_fname = f"{base1}-{base2}_matches.png"
            matches_path = os.path.join(out_matches_dir, matches_fname)
            plot_inlier_matches(img[i], img[j], features[i]["kp"], features[j]["kp"], r.get("good", []), r.get("inliers_mask", []), scale=0.5, savepath=matches_path)
        except Exception:
            pass

        M_shifted, canvas_size, x0, y0 = get_stitch_geometry(img[i], img[j], M)
        img_stitched, x_seam = stitch_image(img[i], img[j], M_shifted, canvas_size, x0, y0, is_mask=False)
        mask_stitched, x_seam = stitch_image(msk[i], msk[j], M_shifted, canvas_size, x0, y0, is_mask=True)

        # ensure output directories exist
        os.makedirs(out_aligned_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)

        # save stitched image (convert RGB->BGR for OpenCV) and mask
        base1 = os.path.splitext(os.path.basename(series_img[i]))[0]
        base2 = os.path.splitext(os.path.basename(series_img[j]))[0]
        fname_img = f"{base1}-{base2}.png"
        fname_mask = f"{base1}-{base2}.png"
        img_out_path = os.path.join(out_aligned_dir, fname_img)
        mask_out_path = os.path.join(out_masks_dir, fname_mask)

        try:
            cv2.imwrite(img_out_path, cv2.cvtColor(img_stitched, cv2.COLOR_RGB2BGR))
        except Exception:
            cv2.imwrite(img_out_path, img_stitched)

        cv2.imwrite(mask_out_path, mask_stitched)
        saved += 1

    return {"uid": uid, "status": "ok", "saved_pairs": saved}


def _worker_align(jobs, results, out_images_dir, out_masks_dir):
    """Worker process: consume series jobs and call process_series."""
    while True:
        job = jobs.get()
        if job == "STOP":
            break
        try:
            # determine per-job output subfolders: keep input substructure
            sub = job.get("subdir", "")
            base_out = Path(out_images_dir)
            # base_out is actually the aligned base directory (aligned/)
            out_aligned = str(base_out / sub / "aligned")
            out_matches = str(base_out / sub / "matches")
            out_masks = str(Path(out_masks_dir) / sub / "masks")
            res = process_series(job["series_img"], job["series_msk_seg"], job["series_msk_det"], out_aligned, out_matches, out_masks)
            results.put((job.get("uid"), True, res))
        except Exception as e:
            results.put((job.get("uid"), False, str(e)))


if __name__ == "__main__":

    # directories
    # BASE_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")
    BASE_DIR = Path(os.environ["SCRATCH"])
    subdir = sys.argv[1]

    dir_img = BASE_DIR / "inference_crops" / subdir 
    dir_msk_seg = BASE_DIR / "predictions" / subdir / "symptoms_seg" / "pred"
    dir_msk_det = BASE_DIR / "predictions" / subdir / "symptoms_det" / "pred"

    # output directories for aligned results
    out_images_dir = BASE_DIR / "aligned" / subdir / "images"
    out_masks_dir = BASE_DIR / "aligned" / subdir / "masks"

    # collect all series for images and masks
    all_series_img = get_series(path_images=os.path.join(dir_img))
    all_series_msk_seg = get_series(path_images=os.path.join(dir_msk_seg))
    all_series_msk_det = get_series(path_images=os.path.join(dir_msk_det))

    # optional: filter series by pattern
    # PATTERN = "CH13063920260606"
    # all_series_img = [s for s in all_series_img if any(PATTERN in os.path.basename(p) for p in s)]
    # all_series_msk = [s for s in all_series_msk if any(PATTERN in os.path.basename(p) for p in s)]

    # build mask lookups by uid for segmentation and detection masks
    mask_lookup_seg = { _uid_from_path(s[0]): s for s in all_series_msk_seg if s }
    mask_lookup_det = { _uid_from_path(s[0]): s for s in all_series_msk_det if s }

    # prepare jobs: include both segmentation and detection mask series (may be empty)
    jobs_list = []
    for series_img in all_series_img:
        if not series_img:
            continue
        uid = _uid_from_path(series_img[0])
        series_msk_seg = mask_lookup_seg.get(uid, [])
        series_msk_det = mask_lookup_det.get(uid, [])
        # skip if neither mask type is available
        if not series_msk_seg and not series_msk_det:
            continue
        jobs_list.append({
            "uid": uid,
            "series_img": series_img,
            "series_msk_seg": series_msk_seg,
            "series_msk_det": series_msk_det,
        })

    if not jobs_list:
        print("No series to process.")
    else:
        m = Manager()
        jobs = m.Queue()
        results_q = m.Queue()
        processes = []

        n_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) - 2)        
        # n_workers = 1
        print(f"Running alignment on {len(jobs_list)} series with {n_workers} workers")

        for _ in range(n_workers):
            p = Process(target=_worker_align, args=(jobs, results_q, out_images_dir, out_masks_dir))
            p.daemon = True
            p.start()
            processes.append(p)

        # enqueue jobs
        for job in jobs_list:
            jobs.put(job)

        # tell workers to stop when done
        for _ in range(n_workers):
            jobs.put("STOP")

        # collect results with progress bar
        total = len(jobs_list)
        received = 0
        with tqdm(total=total, desc="Processing series") as pbar:
            while received < total:
                uid, ok, info = results_q.get()
                received += 1
                if ok:
                    print(f"✓ {uid} ({received}/{total}) saved_pairs={info.get('saved_pairs',0)}")
                else:
                    print(f"✗ {uid}: {info} ({received}/{total})")
                pbar.update(1)

        for p in processes:
            p.join()
    
