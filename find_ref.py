
from pathlib import Path
import colorcorrect_parallel as ccp
import numpy as np
from tqdm import tqdm


# Define paths
DATA_DIR = Path(r"O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/train_img_ESWW009")
output_dir = DATA_DIR / "corrected"
save_dir = DATA_DIR / "control"

image_files = sorted(DATA_DIR.glob("*.JPG"))[:3]

# instantiate detector
# should not require a roi guide, as images are are clearer and under more uniform lighting conditions
input_detector = ccp.ColorCheckerDetector(
    fraction=0.005,
    roi=(4000, 900, 2400, 1600),
    kernel_size=7,
    save_dir=save_dir
)

all_patch_vectors = []
image_names = []
# iterate over all images
for img_path in tqdm(image_files, desc="Processing images", unit="image"):
    # try:
    # detect checker on the input image and export control image
    input_patches, coords, _ = input_detector.detect(
        img_path,
        patch_w=270, patch_h=235,
        plot=True
    )
    all_patch_vectors.append(input_patches.reshape(-1))
    image_names.append(img_path.name)

    # except Exception as e:
    #     print(f"✗ {img_path.name}: {e}")

print(f"✓ Done. Output: {output_dir}")

all_patch_vectors = np.stack(all_patch_vectors)
