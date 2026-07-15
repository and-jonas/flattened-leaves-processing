
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages")
OUTPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40")
INPUT_DIR = Path(r"/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages")
OUTPUT_DIR = Path(r"/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40")
CROP_WIDTH = 8192
CROP_HEIGHT = 2048


def make_inference_crop(img_path):

    # get image
    img = cv2.imread(str(img_path))
    if img is None:
        return f"Could not read {img_path}"
    
    # pre-crop to remove background
    img = img[1250:4750, :]

    B, G, R = cv2.split(img.astype(np.float32))

    # get leaf mask
    leaf_index = R + G - 2 * B
    mask = leaf_index > 20

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
 
    rel = img_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel / "inference_crops"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), crop)

    return None

def main():
    images = list(INPUT_DIR.rglob("*.JPG"))
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(make_inference_crop, images),
            total=len(images)
        ))

if __name__ == "__main__":
    main()
