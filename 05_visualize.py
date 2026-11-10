from leaf.visualization import FlattenedVisualizer
from pathlib import Path
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

dir_to_process = Path(os.environ["SCRATCH"]) / "03_PreDiMix"

subdirs = sorted(
    p for p in (dir_to_process / "inference_crops").iterdir()
    if p.is_dir()
)

logging.info(f"Found {len(subdirs)} subdirectories")

for i, subdir in enumerate(subdirs, 1):
    logging.info(f"Processing {i}/{len(subdirs)}: {subdir.name}")

    vis = FlattenedVisualizer(
        src_root=dir_to_process / "predictions" / subdir.name,
        rgb_root=subdir,
        export_root=dir_to_process / "predictions" / subdir.name,
    )

    vis.visualize()

logging.info("Visualization completed")