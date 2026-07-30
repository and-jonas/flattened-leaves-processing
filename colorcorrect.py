from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np


# ============================================================================
# 1. UTILITY FUNCTIONS (existing + enhanced)
# ============================================================================

def get_dark_mask(img, fraction=0.005, roi=None):
    """Find darkest pixels, optionally within ROI."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if roi is None:
        region = gray
        roi_mask = np.ones(gray.shape, dtype=bool)
    else:
        x, y, w, h = roi
        region = gray[y:y+h, x:x+w]

        roi_mask = np.zeros(gray.shape, dtype=bool)
        roi_mask[y:y+h, x:x+w] = True

    n_pixels = region.size
    n_dark = max(1, int(n_pixels * fraction))

    threshold = np.partition(region.flatten(), n_dark)[n_dark]
    mask = (gray <= threshold) & roi_mask
    # plt.imshow(mask)

    return mask

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
    margin=0.33
):
    """Generate ColorChecker patches from bottom-right anchor."""

    h_img, w_img = img.shape[:2]
    patch_values = []
    patch_coords = []

    for row in range(4):
        for col in range(6):

            # position relative to bottom-right patch
            cx = x0 - (5 - col) * patch_w
            cy = y0 - (3 - row) * patch_h

            # full patch boundaries
            x1 = int(cx - patch_w / 2)
            x2 = int(cx + patch_w / 2)

            y1 = int(cy - patch_h / 2)
            y2 = int(cy + patch_h / 2)

            # clip to image boundaries
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w_img, x2)
            y2c = min(h_img, y2)

            # skip if completely outside image
            if x1c >= x2c or y1c >= y2c:
                continue

            # central sampling region
            dx = int((x2c - x1c) * margin)
            dy = int((y2c - y1c) * margin)

            sx1 = x1c + dx
            sx2 = x2c - dx
            sy1 = y1c + dy
            sy2 = y2c - dy

            sampled = img[sy1:sy2, sx1:sx2]
            patch_id = row * 6 + col

            # Store RGB mean value
            patch_values.append(sampled.mean(axis=(0, 1)))

            # Store metadata for visualization
            patch_coords.append({
                "id": patch_id,
                "row": row,
                "col": col,
                "patch": (x1c, y1c, x2c, y2c),
                "sample": (sx1, sy1, sx2, sy2)
            })

    return np.array(patch_values, dtype=np.float64), patch_coords

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

        mask = get_dark_mask(img, fraction=self.fraction, roi=self.roi)
        mask_proc = post_process_mask(mask, kernel_size=self.kernel_size)
        largest_comp, anchor = get_largest_connected_component(mask_proc)

        patches, coords = generate_checker_grid_from_anchor(
            img, anchor[0], anchor[1], patch_w, patch_h
        )

        # default: save to DATA_DIR / "control" assuming structure
        control_dir = self.save_dir if self.save_dir else img_path.parent.parent / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        save_path = control_dir / f"{img_path.stem}.JPG"

        if plot:
            plot_checker_grid(img, coords, save_path=save_path)

        return patches, coords, anchor


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

        n = len(source_patches)
        if n < 3:
            raise ValueError(f"Need at least 3 patches, got {n}")

        # Fit: [RGB, 1] @ C = RGB_ref  →  C = [A; b]
        X = np.hstack([
            source_patches.astype(np.float32),
            np.ones((n, 1), dtype=np.float32)
        ])
        
        reference = reference_patches.astype(np.float32)
        C, residuals, rank, s = np.linalg.lstsq(X, reference, rcond=None)

        self.A = C[:3, :]
        self.b = C[3, :]

        # Compute RMSE
        pred = X @ C
        self.rmse = np.sqrt(np.mean((pred - reference) ** 2))

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
        pred = self.apply(source_patches).astype(np.float32)
        error = np.abs(pred - reference_patches.astype(np.float32))
        print(f"Validation on {len(source_patches)} patches:")
        print(f"  Mean error per channel: {error.mean(axis=0)}")
        print(f"  Max error per channel:  {error.max(axis=0)}")

# ============================================================================
# 4. BATCH PROCESSING
# ============================================================================

def batch_correct_images(image_dir, transform, output_dir=None, suffix="_corrected"):
    """Apply color correction to all images in a directory."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir) if output_dir else image_dir / "corrected"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.jpg"))

    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_corr = transform.apply(img)
            img_corr_bgr = cv2.cvtColor(img_corr, cv2.COLOR_RGB2BGR)

            out_path = output_dir / f"{img_path.stem}{suffix}.JPG"
            cv2.imwrite(str(out_path), img_corr_bgr)
            print(f"✓ {img_path.name} → {out_path.name}")
        except Exception as e:
            print(f"✗ {img_path.name}: {e}")

    return output_dir


# ============================================================================
# 5. MAIN WORKFLOW (example usage)
# ============================================================================

if __name__ == "__main__":
    
    # Define paths
    DATA_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")
    ref_file = DATA_DIR / "ref" / "20240529_110337_ESWW0090152_5.JPG"
    input_dir = DATA_DIR / "CameraUnknown"
    output_dir = DATA_DIR / "corrected"
    save_dir = DATA_DIR / "control"

    # Step 1: Detect ColorChecker in reference image
    # this needs a roi guide to reliably find the color checker
    # due to complex lighting and many black parts present in the image
    print("\n=== Step 1: Detect reference ColorChecker ===")
    ref_detector = ColorCheckerDetector(
        fraction=0.05,
        roi=(4475, 1100, 1700, 1000),
        kernel_size=7,
        save_dir=save_dir
    )
    ref_patches, _, _ = ref_detector.detect(
        ref_file,
        patch_w=280, patch_h=240,
        plot=True
    )

    # Step 2: Batch process (detect per-image, calibrate to same reference, save control + corrected)
    print(f"\n=== Step 5: Batch correct images in {input_dir} ===")

    output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = save_dir if save_dir else output_dir.parent / "control"
    control_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.JPG")) + sorted(input_dir.glob("*.jpg"))

    # instantiate detector
    # should not require a roi guide, as images are are clearer and under more uniform lighting conditions
    input_detector = ColorCheckerDetector(
        fraction=0.01,
        roi=None,
        kernel_size=7,
        save_dir=save_dir
    )

    # iterate over all images
    for img_path in image_files:
        try:
            # detect checker on the input image and export control image
            input_patches, coords, _ = input_detector.detect(
                img_path,
                patch_w=320, patch_h=270,
                plot=True
            )

            # calibrate transform using this image's patches -> reference patches (constant target)
            transform = ColorTransform()
            transform.calibrate(input_patches, ref_patches)

            # apply transform and save corrected image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_corr = transform.apply(img)
            img_corr_bgr = cv2.cvtColor(img_corr, cv2.COLOR_RGB2BGR)

            out_path = output_dir / f"{img_path.stem}.JPG"
            cv2.imwrite(str(out_path), img_corr_bgr)
            print(f"✓ {img_path.name} → {out_path.name}")

        except Exception as e:
            print(f"✗ {img_path.name}: {e}")

    print(f"✓ Done. Output: {output_dir}")
