
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

INPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages")
OUTPUT_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40")
# INPUT_DIR = Path(r"/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages")
# OUTPUT_DIR = Path(r"/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/WW40")
CROP_WIDTH = 8192
CROP_HEIGHT = 2048
PARALLEL = True

images = list(INPUT_DIR.rglob("*.JPG"))
indices = [i for i, f in enumerate(images) if "102927" in Path(f).name]
img_path = images[indices[1]]

def make_inference_crop(img_path):

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
 
    # export crop
    rel = img_path.relative_to(INPUT_DIR)
    out_path = OUTPUT_DIR / rel.parent / "inference_crops" / (img_path.stem + ".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)

    return None

def main():
    images = list(INPUT_DIR.rglob("*.JPG"))
    if PARALLEL:
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = executor.map(make_inference_crop, images)

            for result in tqdm(
                results,
                total=len(images),
                desc="Processing images"
            ):
                if result is not None:
                    print(result)

    else:
        for img_path in tqdm(
            images,
            total=len(images),
            desc="Processing images"
        ):
            result = make_inference_crop(img_path)
            if result is not None:
                print(result)

if __name__ == "__main__":
    main()
