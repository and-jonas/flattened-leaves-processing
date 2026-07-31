

from pathlib import Path
import colorcorrect_parallel as ccp
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# Define paths
DATA_DIR = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/train_img_ESWW009")
output_dir = DATA_DIR / "corrected"
save_dir = DATA_DIR / "control"

image_files = sorted(DATA_DIR.glob("*.JPG"))
txt_file = DATA_DIR / "exclude_img.txt"
exclude_paths = [
    Path(line.strip().strip('"'))
    for line in txt_file.read_text().splitlines()
    if line.strip()
]
image_files_ref = [p for p in image_files if p not in exclude_paths]
# len(image_files_ref)  # OK

def _process_image(arg):
    """Worker function run in a separate process.
    arg: tuple (img_path_str, save_dir_str)
    Returns: (img_name, patch_vector or None, error_str or None)
    """
    img_path_str, save_dir_str = arg
    img_path = Path(img_path_str)
    save_dir = Path(save_dir_str)

    detector = ccp.ColorCheckerDetector(
        fraction=0.0085,
        roi=(4000, 900, 2400, 1600),
        kernel_size=7,
        save_dir=save_dir,
    )

    try:
        input_patches, coords, _ = detector.detect(
            img_path,
            patch_w=265, patch_h=235,
            plot=True,
        )
        return img_path.name, input_patches.reshape(-1), None
    except Exception as e:
        return img_path.name, None, str(e)


def main():
    all_patch_vectors = []
    image_names = []

    args = [(str(p), str(save_dir)) for p in image_files_ref]

    # Use a process pool so heavy numpy/OpenCV work runs in parallel without GIL limits.
    with ProcessPoolExecutor() as ex:
        for img_name, vec, err in tqdm(ex.map(_process_image, args), total=len(args), desc="Processing images", unit="image"):
            if err:
                print(f"✗ {img_name}: {err}")
                continue
            all_patch_vectors.append(vec)
            image_names.append(img_name)

    if not all_patch_vectors:
        print("No successful detections.")
        return

    print(f"✓ Done. Output: {output_dir}")

    all_patch_vectors = np.stack(all_patch_vectors)

    mean = np.mean(all_patch_vectors, axis=0)
    std = np.std(all_patch_vectors, axis=0)

    scaled = (all_patch_vectors - mean) / (std + 1e-6)

    median_scaled = np.median(scaled, axis=0)

    distances = np.linalg.norm(scaled - median_scaled, axis=1)

    idx = np.argmin(distances)
    print("Representative image:", image_names[idx])


if __name__ == "__main__":
    main()


