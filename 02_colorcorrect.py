from pathlib import Path
from tracemalloc import stop
import cv2
from matplotlib import pyplot as plt
import numpy as np
from multiprocessing import Manager, Process
import os
import pandas as pd


# ============================================================================
# 1. UTILITY FUNCTIONS (existing + enhanced)
# ============================================================================

def get_color_mask(img, color="red", fraction=0.005, roi=None):
    """
    Find the most distinctive pixels of a given color using a fraction-based
    threshold.

    Parameters
    ----------
    img : ndarray
        RGB image.
    color : str
        Color to detect ("red" or "magenta").
    fraction : float
        Fraction of pixels to retain.
    roi : tuple or None
        Optional ROI as (x, y, w, h).

    Returns
    -------
    mask : ndarray
        Binary mask.
    """

    img = img.astype(float)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    total = R + G + B + 1e-6

    if color == "red":
        # Strong red relative to green and blue
        # score = (R / total) * (R - (G + B) / 2)
        score = (R / total) - (G / total + B / total) / 2  # higher specificity (separates from orange patch)

    elif color == "magenta":
        # Strong red + blue, low green
        score = ((R + B) / total) * ((R + B) / 2 - G)

    else:
        raise ValueError(f"Unknown color: {color}")

    if roi is None:
        region = score
        roi_mask = np.ones(score.shape, dtype=bool)
    else:
        x, y, w, h = roi
        region = score[y:y+h, x:x+w]

        roi_mask = np.zeros(score.shape, dtype=bool)
        roi_mask[y:y+h, x:x+w] = True

    n_pixels = region.size
    n_selected = max(1, int(n_pixels * fraction))

    threshold = np.partition(region.flatten(), -n_selected)[-n_selected]

    mask = (score >= threshold) & roi_mask

    return mask.astype(np.uint8)

def post_process_mask(mask, kernel_size=5):
    """Remove noise and fill holes in binary mask."""
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_open = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )
    mask_clean = cv2.morphologyEx(
        mask_open,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )
    return mask_clean

def get_largest_connected_component(mask):
    """Extract largest connected region and its centroid."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    if num_labels < 2:
        raise ValueError("No connected component found in mask")
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8)
    center_x, center_y = centroids[largest_label]

    return largest_component, (center_x, center_y)


def generate_checker_grid_from_anchor(
    img,
    x0,
    y0,
    patch_w,
    patch_h,
    anchor_row,
    anchor_col,
    angle,
    margin=0.4,
):
    """
    Generate ColorChecker patches from an arbitrary anchor while accounting
    for checker rotation.

    Parameters
    ----------
    angle : float
        Checker rotation in radians (CCW).
    """

    h_img, w_img = img.shape[:2]

    c = np.cos(angle)
    s = np.sin(angle)

    patch_values = []
    patch_coords = []

    for row in range(4):
        for col in range(6):

            # Offset in checker coordinates
            dx = (col - anchor_col) * patch_w
            dy = (row - anchor_row) * patch_h

            # Rotate into image coordinates
            rx = dx * c - dy * s
            ry = dx * s + dy * c

            cx = x0 + rx
            cy = y0 + ry

            x1 = int(round(cx - patch_w / 2))
            x2 = int(round(cx + patch_w / 2))
            y1 = int(round(cy - patch_h / 2))
            y2 = int(round(cy + patch_h / 2))

            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w_img, x2)
            y2c = min(h_img, y2)

            if x1c >= x2c or y1c >= y2c:
                patch_values.append(np.array([np.nan, np.nan, np.nan]))
                continue

            dxm = int((x2c - x1c) * margin)
            dym = int((y2c - y1c) * margin)

            sx1 = x1c + dxm
            sx2 = x2c - dxm
            sy1 = y1c + dym
            sy2 = y2c - dym

            sampled = img[sy1:sy2, sx1:sx2]
            n_pix_patch = sampled.shape[0]*sampled.shape[1]
            if n_pix_patch < 2500:
                patch_values.append(np.array([np.nan, np.nan, np.nan]))
            else:
                patch_values.append(sampled.mean(axis=(0, 1)))

            patch_coords.append({
                "id": row * 6 + col,
                "row": row,
                "col": col,
                "patch": (x1c, y1c, x2c, y2c),
                "sample": (sx1, sy1, sx2, sy2),
                "center": (cx, cy),
            })

    return np.asarray(patch_values, dtype=np.float64), patch_coords

def plot_checker_grid(img, patch_coords, save_path=None):

    img_vis = img.copy()

    for p in patch_coords:

        x1, y1, x2, y2 = p["patch"]
        sx1, sy1, sx2, sy2 = p["sample"]

        # full patch
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # sampled area
        cv2.rectangle(img_vis, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

        cv2.putText(img_vis, f'{p["row"]},{p["col"]}',
                    (x1+3, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(img_vis)
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()

    if save_path is not None:
        cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))


# ============================================================================
# 2. COLORCHECKER DETECTION
# ============================================================================

class ColorCheckerDetector:
    """Detect ColorChecker patches in an image."""
    
    def __init__(self, fraction=0.005, roi=None, kernel_size=5, save_dir=None):
        self.fraction = fraction
        self.roi = roi
        self.kernel_size = kernel_size
        self.save_dir = save_dir

    def detect(self, img_path, patch_w=30, patch_h=30, plot=False):
        """
        Detect ColorChecker and extract patch values.
        
        Returns:
            patches (np.ndarray): RGB values, shape (n_patches, 3)
            coords (list): metadata for each patch
            anchor (tuple): (x, y) of anchor point
        """
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get red patch
        mask_red = get_color_mask(img, color = "red", fraction=self.fraction, roi=self.roi)
        # plt.imshow(mask_red)
        # plt.show(block=True)
        mask_red_proc = post_process_mask(mask_red, kernel_size=self.kernel_size)
        largest_comp_red, anchor1 = get_largest_connected_component(mask_red_proc)
        
        # get magenta patch
        mask_mag = get_color_mask(img, color="magenta", fraction=self.fraction, roi=self.roi)
        # plt.imshow(mask_mag)
        # plt.show(block=True)
        mask_mag_proc = post_process_mask(mask_mag, kernel_size=self.kernel_size)
        largest_comp_mag, anchor2 = get_largest_connected_component(mask_mag_proc)

        combined_mask = largest_comp_mag | largest_comp_red

        # fig, axes = plt.subplots(
        #     1, 2,
        #     figsize=(12, 6),
        #     sharex=True,
        #     sharey=True
        # )

        # axes[0].imshow(mask_red)
        # axes[0].set_title("Image 1")

        # axes[1].imshow(img)
        # axes[1].set_title("Image 2")

        # for ax in axes:
        #     ax.axis("off")

        # plt.tight_layout()
        # plt.show()

        dx = anchor2[0] - anchor1[0]
        dy = anchor2[1] - anchor1[1]

        angle = np.arctan2(dy, dx)

        patches, coords = generate_checker_grid_from_anchor(
            img, x0=anchor1[0], y0=anchor1[1], patch_w=patch_w, patch_h=patch_h, anchor_row=2, anchor_col=2,
            angle=angle, margin=0.375
        )

        # default: save to DATA_DIR / "control" assuming structure
        control_dir = self.save_dir if self.save_dir else img_path.parent.parent / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        save_path = control_dir / f"{img_path.stem}.JPG"

        if plot:
            plot_checker_grid(img, coords, save_path=save_path)

        return patches, coords, anchor1


# ============================================================================
# 3. COLOR TRANSFORM CALIBRATION
# ============================================================================

class ColorTransform:
    """Learn & apply affine color correction."""
    
    def __init__(self):
        self.A = None
        self.b = None
        self.rmse = None

    def calibrate(self, source_patches, reference_patches):
        """
        Fit affine transform: RGB_out = RGB_in @ A + b
        
        Parameters:
            source_patches (np.ndarray): shape (n, 3)
            reference_patches (np.ndarray): shape (n, 3)
        """
        if source_patches.shape != reference_patches.shape:
            raise ValueError(
                f"Shape mismatch: source {source_patches.shape} "
                f"vs reference {reference_patches.shape}"
            )

        # Filter out any rows containing NaNs in either source or reference
        valid_mask = (~np.isnan(source_patches).any(axis=1)) & (~np.isnan(reference_patches).any(axis=1))
        valid_idx = np.where(valid_mask)[0]
        n = len(valid_idx)
        if n < 3:
            raise ValueError(f"Need at least 3 valid patches for calibration, got {n}")

        src = source_patches[valid_idx].astype(np.float32)
        ref = reference_patches[valid_idx].astype(np.float32)

        # Fit: [RGB, 1] @ C = RGB_ref  →  C = [A; b]
        X = np.hstack([
            src,
            np.ones((n, 1), dtype=np.float32)
        ])

        C, residuals, rank, s = np.linalg.lstsq(X, ref, rcond=None)

        self.A = C[:3, :]
        self.b = C[3, :]

        # Compute RMSE only on the used (valid) patches
        pred = X @ C
        self.rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))

        return self

    def apply(self, img_array):
        """Apply color correction to an image or patch array."""
        if self.A is None or self.b is None:
            raise RuntimeError("Transform not calibrated. Call calibrate() first.")

        img_float = img_array.astype(np.float32)
        corrected = img_float @ self.A + self.b
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def validate(self, source_patches, reference_patches):
        """Check fit quality on patches."""
        # Filter valid rows
        valid_mask = (~np.isnan(source_patches).any(axis=1)) & (~np.isnan(reference_patches).any(axis=1))
        valid_idx = np.where(valid_mask)[0]
        n = len(valid_idx)
        if n == 0:
            print("Validation: no valid patches to evaluate (all NaN)")
            return

        src = source_patches[valid_idx]
        ref = reference_patches[valid_idx].astype(np.float32)

        pred = self.apply(src).astype(np.float32)
        error = np.abs(pred - ref)
        print(f"Validation on {n} patches (filtered NaNs):")
        print(f"  Mean error per channel: {error.mean(axis=0)}")
        print(f"  Max error per channel:  {error.max(axis=0)}")

# ============================================================================
# 4. BATCH PROCESSING
# ============================================================================

def _worker_process(jobs, results, ref_patches, fraction, roi, kernel_size, patch_w, patch_h):
    """Top-level worker for multiprocessing (picklable)."""
    while True:
            job = jobs.get()
            if job == "STOP":
                break
            img_name = job["image_name"]
            img_path = Path(job["image_path"])
            # output dir
            parts = list(img_path.parent.parts)
            idx = parts.index("renamed")
            parts[idx:idx + 1] = ["colorcorrected", "corrected"]
            output_dir = Path(*parts)
            # ensure the per-image output directory exists (match control structure)
            output_dir.mkdir(parents=True, exist_ok=True)
            # control dir
            parts = list(img_path.parent.parts)
            idx = parts.index("renamed")
            parts[idx:idx + 1] = ["colorcorrected", "control"]
            control_dir = Path(*parts)
            # instantiate detector with the per-image control directory
            detector = ColorCheckerDetector(fraction=fraction, roi=roi, kernel_size=kernel_size, save_dir=control_dir)
            try:
                input_patches, coords, _ = detector.detect(
                    img_path,
                    patch_w=patch_w, patch_h=patch_h,
                    plot=True
                )
                transform = ColorTransform()
                transform.calibrate(input_patches, ref_patches)

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    raise FileNotFoundError(f"Cannot read {img_path}")

                out_path = output_dir / f"{img_path.stem}.png"

                # If RMSE is large, keep the original image instead of the corrected one
                if (transform.rmse is not None) and (transform.rmse > 20):
                    # save original BGR image
                    cv2.imwrite(str(out_path), img_bgr)
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    img_corr = transform.apply(img_rgb)
                    img_corr_bgr = cv2.cvtColor(img_corr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_path), img_corr_bgr)

                results.put((img_name, True, transform.rmse))
            except Exception as e:
                results.put((img_name, False, str(e)))


# ============================================================================
# 5. MAIN WORKFLOW (example usage)
# ============================================================================

if __name__ == "__main__":
    
    # # Define paths: local
    # REF_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/ref")
    # BASE_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")
    # INPUT_DIR = BASE_DIR / "renamed"
    # OUTPUT_DIR = BASE_DIR / "colorcorrected"

    # Define paths: SLURM
    BASE_DIR = Path(os.environ["SCRATCH"])
    REF_DIR = BASE_DIR / "ref"
    INPUT_DIR = BASE_DIR / "renamed"
    OUTPUT_DIR = BASE_DIR / "colorcorrected"

    ref_file = REF_DIR / "20240619_092442_ESWW0090037_2.JPG"
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR / "corrected"
    save_dir = OUTPUT_DIR / "control"

    # Step 1: Detect ColorChecker in reference image
    # this needs a roi guide to reliably find the color checker
    print("\n=== Step 1: Detect reference ColorChecker ===")
    ref_detector = ColorCheckerDetector(
        fraction=0.05,
        roi=(4475, 1100, 1700, 1000),
        kernel_size=7,
        save_dir=save_dir
    )
    ref_patches, _, _ = ref_detector.detect(
        ref_file,
        patch_w=270, patch_h=235,
        plot=True
    )

    # Step 2: Batch process (detect per-image, calibrate to same reference, save control + corrected)
    print(f"\n=== Step 2: Batch correct images in {input_dir} ===")

    output_dir.mkdir(parents=True, exist_ok=True)

    # list all files to process (recursively)
    image_files = sorted(input_dir.glob("**/*.JPG"))

    # there seem to be duplicates; remove duplicates by name (keep first occurrence)
    seen = set()
    image_files_unique = []
    for f in image_files:
        if f.name not in seen:
            image_files_unique.append(f)
            seen.add(f.name)
    image_files = image_files_unique

    # filter out already processed images
    skip_processed = False
    if skip_processed:
        existing = set(p.stem for p in output_dir.glob("**/*.png"))
        image_files = [f for f in image_files if f.stem not in existing]

    # # select problematic examples
    # keep_names = [
    #     "20260628_181329_CH13063920260402.JPG",
    #     "20260625_113946_CH13063920260797.JPG",
    #     "20260624_170248_CH13063920260712.JPG",
    #     "20260626_150808_CH13063920260439.JPG",
    #     "20260630_150126_CH13063920260056.JPG",
    #     "20260626_142815_CH13063920260416.JPG"
    # ]
    # image_files = [f for f in image_files if f.name in keep_names]

    # instantiate detector parameters
    fraction = 0.015 ## CRITICAL PARAMETER: fraction of pixels to consider for patch detection. Ensure that as much of the patch is detected as possible; without spurious detection of similar patches
    roi = (0, 0, 2000, 2000)  # upper left corner
    kernel_size = 11
    pw, ph = 320, 270

    # Build job/result queues and spawn workers
    m = Manager()
    jobs = m.Queue()
    results = m.Queue()
    processes = []

    max_jobs = len(image_files)
    if max_jobs == 0:
        print("No images to process.")
    else:
        # enqueue jobs
        for img_path in image_files:
            print(f"[Main] queueing {img_path.stem}")
            jobs.put({
                "image_name": img_path.stem,
                "image_path": str(img_path)
            })

        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))  # when processing on Gamarello
        print(f"running on {input_dir} with {n_workers} workers ===")
        for _ in range(n_workers):
            p = Process(
                target=_worker_process,
                args=(
                    jobs,
                    results,
                    ref_patches,
                    fraction,
                    roi,
                    kernel_size,
                    pw,
                    ph
                )
            )
            p.daemon = True
            p.start()
            processes.append(p)
            jobs.put("STOP")

        # collect results and write RMSE continuously
        rmse_log = []

        # prepare RMSE CSV for incremental writing
        save_dir.mkdir(parents=True, exist_ok=True)
        rmse_file = save_dir / "rmse_log.csv"
        write_header = not rmse_file.exists()
        if write_header:
            with open(rmse_file, "w", newline="") as fh:
                fh.write("image,ok,rmse,error\n")

        count = 0
        while count < max_jobs:
            name, ok, info = results.get()
            count += 1
            if ok:
                rmse = info
                rmse_log.append((name, rmse))
                with open(rmse_file, "a", newline="") as fh:
                    fh.write(f"{name},True,{rmse},\n")
                print(f"✓ {name} ({count}/{max_jobs})  RMSE={rmse:.2f}")
            else:
                # write error row with empty RMSE
                with open(rmse_file, "a", newline="") as fh:
                    # escape quotes in error string
                    err = str(info).replace('"', '""')
                    fh.write(f"{name},False,,\"{err}\"\n")
                print(f"✗ {name}: {info} ({count}/{max_jobs})")

        for p in processes:
            p.join()

    print(f"✓ Done. Output: {output_dir}")
