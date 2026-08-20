# ======================================================================================================================
# Makes 2048 x 8192 px crops of leaf images for inference
# ======================================================================================================================

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import os

# directories (local)
# PARENT_INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages")
# PARENT_OUTPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/06_WW40/LeafImages")
# PARENT_INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/02_CHWW001/1260/LeafImages/renamed")
# PARENT_OUTPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/02_CHWW001/1260/LeafImages/inference_crops")
# PARENT_INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/corrected")
# PARENT_OUTPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/renamed")
# PARENT_INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/colorcorrected/corrected")
# PARENT_OUTPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf/inference_crops")

# directories (server)
BASE_DIR = Path(os.environ["SCRATCH"])
PARENT_INPUT_DIR = BASE_DIR / "06_WW040/LeafImages/1260/renamed"
PARENT_OUTPUT_DIR = BASE_DIR / "06_WW040/LeafImages/1260/inference_crops"

CROP_WIDTH = 8192
CROP_HEIGHT = 2048
PARALLEL = True

def make_inference_crop(args):
    img_path, input_dir, output_dir = args

    # get image
    img = cv2.imread(str(img_path))
    if img is None:
        return f"Could not read {img_path}"
    
    # check orientation and rotate if needed
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # pre-crop to remove background
    img = img[1250:4750, :]

    # get leaf mask
    B, G, R = cv2.split(img.astype(np.float32))
    leaf_index = R + G - 2 * B

    h, w = leaf_index.shape

    # Right 2/3 of the image
    x_start = w // 3
    leaf_index_roi = leaf_index[:, x_start:]

    # Lower threshold until we find a connected component with area close to 1.5 Mpx
    best_score = np.inf
    best_threshold = None

    for threshold in np.arange(20, -80, -20):
        mask_roi = leaf_index_roi > threshold

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_roi.astype(np.uint8), connectivity=8
        )

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]

            # Target leaf size = 1.5 Mpx
            score = abs(area - 1_500_000)

            if score < best_score:
                best_score = score
                best_threshold = threshold

    # Apply the selected threshold to the right 2/3
    mask_roi = leaf_index_roi > best_threshold

    # Keep only the largest connected component
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_roi.astype(np.uint8), connectivity=8
    )

    if n > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_roi = (labels == largest_component).astype(np.uint8)
    else:
        mask_roi = np.zeros_like(mask_roi, dtype=np.uint8)

    # Put the ROI mask back into the full image
    mask = np.zeros_like(leaf_index, dtype=np.uint8)
    mask[:, x_start:] = mask_roi

    # get controid of foreground pixels
    ys, xs = np.where(mask)
    if len(xs) < 100:
        return f"No mask: {img_path}"
    coords = np.column_stack([xs, ys])
    pca = PCA(n_components=2)
    pca.fit(coords)
    _, cy = pca.mean_

    # crop image around centroid height
    img_h, img_w = img.shape[:2]

    x1 = img_w // 2 - CROP_WIDTH // 2
    x2 = x1 + CROP_WIDTH

    y1 = int(cy - CROP_HEIGHT / 2)
    y1 = max(0, min(y1, img_h - CROP_HEIGHT))
    y2 = y1 + CROP_HEIGHT

    crop = img[y1:y2, x1:x2]
 
    # export crop
    rel = img_path.relative_to(input_dir)
    out_path = output_dir / rel.parent / (img_path.stem + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)

    return None

def main():
    patterns = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png", "*.PNG")
    # collect directories that contain jpeg images
    jpeg_dirs = []

    # include the parent input dir itself if it contains images
    if any(next(PARENT_INPUT_DIR.glob(pat), None) is not None for pat in patterns):
        jpeg_dirs.append(PARENT_INPUT_DIR)

    # include immediate subdirectories (not recursive) that contain images
    jpeg_dirs.extend(
        d
        for d in PARENT_INPUT_DIR.iterdir()
        if d.is_dir()
        and any(next(d.glob(pat), None) is not None for pat in patterns)
    )

    tasks = [
        (img_path, d, PARENT_OUTPUT_DIR / d.relative_to(PARENT_INPUT_DIR))
        for d in jpeg_dirs
        for img_path in {p for pat in patterns for p in d.glob(pat)}
    ]

    print(f"Found {len(tasks)} images in {len(jpeg_dirs)} directories.")

    if PARALLEL:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = executor.map(make_inference_crop, tasks)
            for result in tqdm(results, total=len(tasks), desc="Processing images"):
                if result is not None:
                    print(result)
    else:
        for args in tqdm(tasks, total=len(tasks), desc="Processing images"):
            result = make_inference_crop(args)
            if result is not None:
                print(result)

if __name__ == "__main__":
    main()
