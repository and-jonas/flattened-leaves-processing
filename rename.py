# ======================================================================================================================
# Rename images of single leaves using datetime of image capture from EXIF
# and QR code in image.
# ======================================================================================================================

import concurrent.futures
from pathlib import Path
import shutil
import cv2
import os
import exifread
from pyzbar.pyzbar import decode
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
from qreader import QReader

#instantiate QR reader
qreader = QReader()

parent_dir = Path(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages/1510"
)

MAX_WIDTH = 3200
WORKERS = min(8, (os.cpu_count() or 4))
PARALLEL = True          # Set to False for sequential processing, True for parallel


def extract_datetime_original(file_path: Path) -> str:
    with file_path.open("rb") as f:
        tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
    dtime_tag = tags.get("EXIF DateTimeOriginal")
    if not dtime_tag:
        return file_path.stem
    dtime = str(dtime_tag.values)
    dtime = dtime.replace(":", "")
    return dtime.replace(" ", "_")


def decode_barcode(file_path: Path) -> str:
    """
    Search for QR code within the white label region.
    Returns file stem if no QR code is found.
    """
    img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)

    if img is None:
        return file_path.stem

    # Create a QReader instance
    decoded_text = qreader.detect_and_decode(image=img)
    if decoded_text and decoded_text[0]:
        return decoded_text[0]   
    return file_path.stem

def make_unique_file(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    counter = 1
    while True:
        candidate = dst.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def process_file(file_path: Path, output_dir: Path) -> tuple[Path, Path]:
    qr_id = decode_barcode(file_path)
    date_stamp = extract_datetime_original(file_path)
    new_name = f"{date_stamp}_{qr_id}.JPG"
    dst = output_dir / new_name
    dst = make_unique_file(dst)
    shutil.copy2(file_path, dst)
    return file_path, dst


def process_directory(directory: Path) -> None:
    jpeg_files = list(directory.glob("*.JPG"))[:32]
    if not jpeg_files:
        return

    # Create renamed sub-directory
    renamed_dir = directory / "renamed"
    renamed_dir.mkdir(exist_ok=True)

    results = []
    
    if PARALLEL:
        # Parallel processing with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_file, p, renamed_dir) for p in jpeg_files]
            
            if tqdm is not None:
                with tqdm(total=len(jpeg_files), desc=f"Processing {directory.name}", unit="file") as pbar:
                    for fut in concurrent.futures.as_completed(futures):
                        res = fut.result()
                        results.append(res)
                        pbar.update(1)
            else:
                for fut in concurrent.futures.as_completed(futures):
                    results.append(fut.result())
    else:
        # Sequential processing
        for file_path in tqdm(jpeg_files, desc=f"Processing {directory.name}", unit="file"):
            result = process_file(file_path, renamed_dir)
            results.append(result)

    log_path = directory / "renaming_log.csv"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("old_names,new_names\n")
        for old_path, new_path in results:
            log_file.write(f"{old_path},{new_path}\n")


if __name__ == "__main__":
    patterns = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg")
    jpeg_dirs = [
        d
        for d in parent_dir.rglob("*")
        if d.is_dir()
        and any(next(d.glob(pat), None) is not None for pat in patterns)
    ]
    for jpeg_dir in jpeg_dirs:
        process_directory(jpeg_dir)

