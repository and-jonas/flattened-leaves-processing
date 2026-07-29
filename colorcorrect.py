from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import numpy as np


PARENT_INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")
file_path = PARENT_INPUT_DIR / "CameraUnknown" / "20260625_081007_CH13063920260750.JPG"
ref_file_path = PARENT_INPUT_DIR / "ref" / "20240529_110337_ESWW0090152_5.JPG"

import numpy as np
import cv2


import numpy as np
import cv2


def get_dark_mask(img, fraction=0.005, roi=None):
    """
    Find darkest pixels.

    Parameters
    ----------
    img : np.ndarray
        RGB image.
    fraction : float
        Fraction of pixels considered dark (e.g. 0.005 = 0.5%).
    roi : tuple or None
        Optional ROI as (x, y, w, h).

    Returns
    -------
    mask : np.ndarray
        Binary mask with same size as image.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # select region for threshold calculation
    if roi is None:
        region = gray
        roi_mask = np.ones(gray.shape, dtype=bool)
    else:
        x, y, w, h = roi
        region = gray[y:y+h, x:x+w]

        roi_mask = np.zeros(gray.shape, dtype=bool)
        roi_mask[y:y+h, x:x+w] = True

    # number of pixels to include
    n_pixels = region.size
    n_dark = max(1, int(n_pixels * fraction))

    # intensity threshold corresponding to darkest pixels
    threshold = np.partition(
        region.flatten(),
        n_dark
    )[n_dark]

    # create mask
    mask = (gray <= threshold) & roi_mask

    return mask

def post_process_mask(mask, kernel_size=5):
    """
    Post-process a binary mask to remove small objects and fill holes.

    mask: binary mask (0 or 1)
    kernel_size: size of the morphological kernel
    """

    # ensure binary uint8 mask
    mask = (mask > 0).astype(np.uint8)

    # define kernel size (adjust depending on noise size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # remove small objects
    mask_open = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # fill small gaps/holes
    mask_clean = cv2.morphologyEx(
        mask_open,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask_clean

def get_largest_connected_component(mask):
    """
    Get the largest connected component in a binary mask.

    mask: binary mask (0 or 1)
    """

    # find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    # ignore background (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # keep only largest component
    largest_component = (labels == largest_label).astype(np.uint8)

    # centroid of largest component
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
    """
    Generate ColorChecker Nano patches from bottom-right anchor.
    Clips patches at image borders.

    margin:
        fraction of patch excluded on each side for sampling
    """

    h_img, w_img = img.shape[:2]

    patches = []
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

            sampled = img[
                sy1:sy2,
                sx1:sx2
            ]

            patches.append(
                sampled.mean(axis=(0,1))
            )

            patch_coords.append({
                "row": row,
                "col": col,
                "patch": (x1c, y1c, x2c, y2c),
                "sample": (sx1, sy1, sx2, sy2)
            })

    return np.array(patches), patch_coords

def plot_checker_grid(img, patch_coords):

    img_vis = img.copy()

    for p in patch_coords:

        x1, y1, x2, y2 = p["patch"]
        sx1, sy1, sx2, sy2 = p["sample"]

        # full patch
        cv2.rectangle(
            img_vis,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # sampled area
        cv2.rectangle(
            img_vis,
            (sx1, sy1),
            (sx2, sy2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img_vis,
            f'{p["row"]},{p["col"]}',
            (x1+3, y1+15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255,0,0),
            1
        )

    plt.figure(figsize=(8,8))
    plt.imshow(img_vis)
    plt.axis("off")
    plt.show()

def process_image(file_path, fraction=0.005, roi=None, kernel_size=5, patch_w=30, patch_h=30, plot=False):
    """
    Process a single image to extract ColorChecker patches.

    Returns a dict with results.
    """

    img = cv2.imread(str(file_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    black_mask = get_dark_mask(img, fraction=fraction, roi=roi)
    black_mask_proc = post_process_mask(black_mask, kernel_size=kernel_size)
    largest_component, (center_x, center_y) = get_largest_connected_component(black_mask_proc)

    # generate checker grid
    values, coords = generate_checker_grid_from_anchor(
        img,
        center_x,
        center_y,
        patch_w,
        patch_h,
        margin=0.33
    )

    if plot:
        plot_checker_grid(img, coords)

    return values

ref_values = process_image(
    ref_file_path, 
    fraction=0.05, 
    roi=(4475, 1100, 1700, 1000), 
    kernel_size=7, 
    patch_w=280, patch_h=240, 
    plot=True
)


values = process_image(file_path, fraction=0.005, roi=(4475, 1100, 1700, 1000), kernel_size=7, patch_w=325, patch_h=275, plot=True)







# get gray values and estimate channel gain
n = 4 if len(values) >= 20 else 5
gray_patches = values[-n:]
mean_gray = gray_patches.mean(axis=0)
print(mean_gray)
target = mean_gray.mean()
gain = target / mean_gray
print(gain)


img_float = img.astype(np.float32)
img_corrected = img_float * gain
img_corrected = np.clip(
    img_corrected,
    0,
    255
).astype(np.uint8)


fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(img_corrected)
axes[1].set_title("Corrected")
axes[1].axis("off")

plt.tight_layout()
plt.show()