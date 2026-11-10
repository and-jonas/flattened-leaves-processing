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
# dir_to_process = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")

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

    vis.visualize(parallel=True, max_workers=4)

logging.info("Visualization completed")