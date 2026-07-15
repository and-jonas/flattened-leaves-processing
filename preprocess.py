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
from tqdm import tqdm
import pytesseract


parent_dir = Path(
    "O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/06_WW40/LeafImages/1510"
)

MAX_WIDTH = 3200
WORKERS = min(8, (os.cpu_count() or 4))
WHITE_LABEL_THRESHOLD = 200  # Threshold for detecting white regions (0-255)
MIN_LABEL_WIDTH = 500        # Minimum width of labeled region
MIN_LABEL_HEIGHT = 100       # Minimum height of labeled region


def find_white_label_region(img: cv2.Mat) -> tuple:
    """
    Detect the white label region in the image.
    Returns (y_start, y_end, x_start, x_end) of the label region, or None if not found.
    """
    h, w = img.shape[:2]
    
    # Create binary image of white regions
    white_mask = cv2.inRange(img, WHITE_LABEL_THRESHOLD, 255)
    
    # Find contours of white regions
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest white region
    largest_contour = None
    largest_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, contour_w, contour_h = cv2.boundingRect(contour)
        
        # Filter by size constraints
        if contour_w >= MIN_LABEL_WIDTH and contour_h >= MIN_LABEL_HEIGHT and area > largest_area:
            largest_area = area
            largest_contour = (x, y, contour_w, contour_h)
    
    if largest_contour is None:
        return None
    
    x, y, label_w, label_h = largest_contour
    # Add padding around detected region
    padding = 20
    y_start = max(0, y - padding)
    y_end = min(h, y + label_h + padding)
    x_start = max(0, x - padding)
    x_end = min(w, x + label_w + padding)
    
    return (y_start, y_end, x_start, x_end)


def extract_datetime_original(file_path: Path) -> str:
    with file_path.open("rb") as f:
        tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
    dtime_tag = tags.get("EXIF DateTimeOriginal")
    if not dtime_tag:
        return file_path.stem
    dtime = str(dtime_tag.values)
    dtime = dtime.replace(":", "")
    return dtime.replace(" ", "_")


def read_label_text(file_path: Path, img: cv2.Mat) -> str:
    """
    Extract and read text from the detected white label region.
    Uses automatic white region detection instead of fixed coordinates.
    """
    label_region = find_white_label_region(img)
    
    if label_region is None:
        # Fall back to file stem if no white label found
        return file_path.stem
    
    y_start, y_end, x_start, x_end = label_region
    text_region = img[y_start:y_end, x_start:x_end]
    
    # OCR to extract text from the white label region
    text = pytesseract.image_to_string(text_region).strip()
    return text if text else file_path.stem


def decode_barcode(file_path: Path) -> str:
    """
    Read text from white label region in image.
    Falls back to file stem if no label is found.
    """
    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return file_path.stem
    
    # Read text from the white label region
    return read_label_text(file_path, img)


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
    jpeg_files = list(directory.glob("*.JPG"))
    if not jpeg_files:
        return

    # Create renamed sub-directory
    renamed_dir = directory / "renamed"
    renamed_dir.mkdir(exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        # submit tasks so we can track completions
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

