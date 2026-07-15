"""Scan a folder of images, decode QR codes, and rename files.

Usage examples:
  python preprocess.py --folder images --dry-run
  python preprocess.py --folder images --pattern "{id}_{species}{ext}"

Notes:
  - Requires `opencv-python` (install with `pip install opencv-python`).
  - By default new filename is the raw QR text (sanitized) plus the original extension.
  - If QR text is JSON, you can use keys in the `--pattern` format string.
  - If a file would overwrite an existing file, a numeric suffix is appended.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import cv2


def sanitize_filename(name: str) -> str:
    name = name.strip()
    # replace whitespace with underscore
    name = re.sub(r"\s+", "_", name)
    # remove characters that are invalid in filenames
    name = re.sub(r"[^A-Za-z0-9_.-]", "", name)
    # collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    if not name:
        return "unnamed"
    return name


def parse_qr_text(text: str) -> Dict[str, str]:
    """Try to parse QR text into a dict for formatting.

    - If JSON: return its keys plus `data` containing original text.
    - If lines of key=value: parse them.
    - Otherwise return {'data': text}.
    """
    ctx: Dict[str, str] = {}
    if not text:
        return {"data": ""}
    # try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                ctx[str(k)] = str(v)
            ctx["data"] = text
            return ctx
    except Exception:
        pass

    # try key=value lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kv_found = False
    for ln in lines:
        if "=" in ln:
            kv_found = True
            k, v = ln.split("=", 1)
            ctx[k.strip()] = v.strip()
    if kv_found:
        ctx["data"] = text
        return ctx

    # fallback
    ctx["data"] = text
    return ctx


def make_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    base = dst.stem
    ext = dst.suffix
    i = 1
    while True:
        candidate = dst.with_name(f"{base}_{i}{ext}")
        if not candidate.exists():
            return candidate
        i += 1


def decode_qr_from_image(path: Path) -> List[str]:
    img = cv2.imread(str(path))
    if img is None:
        return []
    detector = cv2.QRCodeDetector()
    # detectAndDecodeMulti is available in newer OpenCV; fall back to single
    try:
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)
        if retval and decoded_info:
            return [d for d in decoded_info if d]
    except Exception:
        pass
    # single
    data, points, _ = detector.detectAndDecode(img)
    if data:
        return [data]
    return []


def process_folder(folder: Path, pattern: str, dry_run: bool = True, recursive: bool = False) -> None:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = list(folder.rglob("*") if recursive else folder.iterdir())
    for p in files:
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        qr_texts = decode_qr_from_image(p)
        if not qr_texts:
            print(f"[SKIP] No QR in: {p.name}")
            continue
        # prepare context
        ctx = {}
        # add dataN keys and first `data`
        for idx, t in enumerate(qr_texts):
            ctx[f"data{idx}"] = t
        ctx["data"] = qr_texts[0]
        # merge parsed values if first QR is structured
        parsed = parse_qr_text(qr_texts[0])
        ctx.update(parsed)
        ctx["ext"] = p.suffix

        # attempt formatting
        try:
            new_name = pattern.format(**ctx)
        except Exception:
            # fallback to raw data
            new_name = ctx.get("data", "unnamed") + p.suffix

        new_name = sanitize_filename(new_name)
        if not new_name.lower().endswith(p.suffix.lower()):
            new_name = new_name + p.suffix

        dst = p.with_name(new_name)
        dst = make_unique_path(dst)
        if dry_run:
            print(f"[DRY] {p.name} -> {dst.name}")
        else:
            print(f"[RENAME] {p.name} -> {dst.name}")
            p.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename images using QR code contents")
    parser.add_argument("--folder", "-f", required=True, help="Folder containing images")
    parser.add_argument("--pattern", "-p", default="{data}", help="Filename pattern using fields from QR (default '{data}'). Use {ext} for extension")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without renaming")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}")
        return

    process_folder(folder, args.pattern, dry_run=args.dry_run, recursive=args.recursive)


if __name__ == "__main__":
    main()
