
import importlib

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import utils.base as base_utils
importlib.reload(base_utils)
import os
from scipy import ndimage as ndi
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import traceback
from matplotlib.colors import Normalize

def process_item(args):
    seg_p, det_p, rgb_p, aligned, leaf_data_path, lesion_data_path, out_path = args
    # convert string paths to Path in worker process (needed on Windows spawn)
    leaf_data_path = Path(leaf_data_path)
    lesion_data_path = Path(lesion_data_path)
    out_path = Path(out_path)
    try:
        seg_path = Path(seg_p)
        stem = seg_path.stem
        csv_name = stem + ".csv"
        jpg_name = stem + ".JPG"

        seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        if seg_mask is None:
            print(f"Cannot read segmentation mask: {seg_path}")
            return

        if det_p is not None:
            det_mask = cv2.imread(str(det_p), cv2.IMREAD_UNCHANGED)
        else:
            det_mask = None
        if det_mask is None:
            det_mask = np.zeros_like(seg_mask)

        rgb_path = Path(rgb_p)
        rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"Cannot read RGB image: {rgb_path}")
            return
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        if not aligned:
            det_mask = np.where(det_mask == 0, det_mask, det_mask + 4)
            mask = np.where(det_mask == 0, seg_mask, det_mask)
        else:
            mask = seg_mask

        mask_leaf = np.where((mask >= 1) & (mask != 3), 1, 0).astype("uint8")
        l_mask = np.where(mask == 2, 255, 0).astype("uint8")
        l_mask = base_utils.filter_objects_size(mask=l_mask, size_th=500, dir="smaller")

        kernel = np.ones((3, 3), np.uint8)
        l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_ERODE, kernel, iterations=2)
        l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_ERODE, kernel, iterations=1)
        l_mask = cv2.morphologyEx(l_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        l_mask = np.where(l_mask, 255, 0).astype("uint8")

        # ----------------------------------------------------------------------------
        # leaf statistics
        # ----------------------------------------------------------------------------

        la_tot = len(np.where(mask != 0)[0])
        la_damaged = len(np.where((mask != 0) & (mask != 1))[0])
        la_healthy = len(np.where(mask == 1)[0])
        la_damaged_f = la_damaged / la_tot if la_tot else 0
        la_healthy_f = la_healthy / la_tot if la_tot else 0
        la_insect = len(np.where(mask == 3)[0])
        la_insect_f = la_insect / la_tot if la_tot else 0
        n_pycn = len(np.where(mask == 5)[0])
        rust_idx = np.where(mask == 6)
        rust_point_list = base_utils.filter_points(x=rust_idx[1], y=rust_idx[0], min_distance=7)
        n_rust = len(rust_point_list)
        contours, _ = cv2.findContours(l_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        n_lesion = len(contours)
        placl = len(np.where((mask == 2) | (mask == 5))[0]) / (la_tot - la_insect) if (la_tot - la_insect) > 0 else 0
        pycn_density = n_pycn / (la_tot - la_insect) if (la_tot - la_insect) > 0 else 0
        rust_density = n_rust / (la_tot - la_insect) if (la_tot - la_insect) > 0 else 0

        # get pcnidia density
        pycn_distmap, pycn_dmap, norm_d = base_utils.get_pycnidia_maps(
            mask=mask,
            resize_factor=5, bandwidth=25, kernel='gaussian'
        )

        # get pycnidiation area
        thresholds = [0.0001, 0.0005, 0.001]
        plap = {}
        for t in thresholds:
            plap_mask = (l_mask) & (np.where(norm_d >= t, 1, 0))
            plap[t] = len(np.where(plap_mask)[0]) / (la_tot - la_insect) if (la_tot - la_insect) > 0 else 0

        # Convert density to RGB colormap
        density_vis = Normalize(vmin=0,vmax=0.005, clip=True)(norm_d)
        overlay = rgb_img.copy()
        density_color = plt.cm.plasma(density_vis)[:, :, :3]
        density_color = (density_color * 255).astype(np.uint8)

        # Overlay on original RGB image
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1 - alpha, density_color, alpha, 0)
        cv2.imwrite(str(out_path / "pmap" / jpg_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        leaf_data = [{
            'la_tot': la_tot,
            'la_damaged': la_damaged,
            'la_healthy': la_healthy,
            'la_damaged_f': la_damaged_f,
            'la_healthy_f': la_healthy_f,
            'la_insect': la_insect,
            'la_insect_f': la_insect_f,
            'n_pycn': n_pycn,
            'n_rust': n_rust,
            'n_lesion': n_lesion,
            'placl': placl,
            'plap@0.0001': plap[0.0001],
            'plap@0.0005': plap[0.0005],
            'plap@0.001': plap[0.001],
            'pycn_density': pycn_density,
            'rust_density': rust_density
        }]

        # ensure output subdirs
        for p in [leaf_data_path, lesion_data_path, out_path / "pmap"]:
            p.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(leaf_data).to_csv(leaf_data_path / csv_name, index=False)

        # ----------------------------------------------------------------------------
        # lesion analysis
        # ----------------------------------------------------------------------------

        lesion_data = []
        instance_mask = np.zeros_like(mask)
        for idx, contour in enumerate(contours):
            rect = cv2.boundingRect(contour)
            roi = base_utils.select_roi_2(rect=rect, mask=l_mask)
            instance_mask = np.where(roi, idx + 1, instance_mask)
            rect = base_utils.get_bounding_boxes(rect=rect)

            contour_area = cv2.contourArea(contour)
            if contour_area <= 0:
                continue
            contour_perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull) if hull is not None else 1
            contour_solidity = float(contour_area) / hull_area if hull_area else 0
            _, _, w, h = cv2.boundingRect(contour)

            pycn_mask = np.where(roi, mask, 0)
            n_pycn_loc = len(np.where(pycn_mask == 5)[0])
            pycn_density_lesion = n_pycn_loc / contour_area

            rust_mask = np.where(roi, mask, 0)
            n_rust_loc = len(np.where(rust_mask == 6)[0])
            rust_density_lesion = n_rust_loc / contour_area

            ldat = {'area': contour_area,
                    'perimeter': contour_perimeter,
                    'solidity': contour_solidity,
                    'max_width': w,
                    'max_height': h,
                    'n_pycn': n_pycn_loc,
                    'pycn_density_lesion': pycn_density_lesion,
                    'rust_density_lesion': rust_density_lesion
                    }

            # ldat = ldat | pycn_features
            lesion_data.append(ldat)

        if lesion_data:
            df_les = pd.DataFrame(lesion_data, columns=lesion_data[0].keys())
            df_les.to_csv(lesion_data_path / csv_name, index=False)
        else:
            pd.DataFrame(columns=['area', 'perimeter']).to_csv(lesion_data_path / csv_name, index=False)

        print(f"Processed {stem}")

    except Exception:
        # log exception to per-run errors.log and continue processing other items
        try:
            err_log = out_path / "errors.log"
            with err_log.open("a", encoding="utf-8") as f:
                f.write(f"ERROR processing {seg_p}\n")
                traceback.print_exc(file=f)
                f.write("\n")
        except Exception:
            # fallback to printing if logging fails
            print(f"\nERROR processing {seg_p}", flush=True)
            traceback.print_exc()
        return

def main(aligned=False):
    # map seg/det/rgb by stem
    seg_map = {p.stem: p for p in seg_masks}
    det_map = {p.stem: p for p in det_masks}
    rgb_map = {p.stem: p for p in rgb_imgs}

    tasks = []
    for stem, seg_p in seg_map.items():
        det_p = det_map.get(stem)
        rgb_p = rgb_map.get(stem)
        if rgb_p is None:
            print(f"No RGB for {stem}, skipping")
            continue
        tasks.append((
            str(seg_p),
            str(det_p) if det_p is not None else None,
            str(rgb_p),
            aligned,
            str(leaf_data_path),
            str(lesion_data_path),
            str(out_path),
        ))  # include output paths so workers on Windows have them

    if not tasks:
        print("No valid tasks to process.")
        return

    n_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) - 2)  # leave 2 CPUs free
    with Pool(processes=n_workers) as pool:
        if tqdm is not None:
            for _ in tqdm(pool.imap_unordered(process_item, tasks), total=len(tasks), desc="Processing", unit="item"):
                pass
        else:
            # fallback: no progress bar
            pool.map(process_item, tasks)


if __name__ == "__main__":

    base_path = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/E_Work/03_PreDiMix/Uitikon/20260623_Uitikon_Leaf")
    # base_path = Path(os.environ["SCRATCH"])

    aligned = False  # set to False if using individual images instead of aligned images

    if not aligned:
        seg_masks = list((base_path / "predictions").glob("*/symptoms_seg/pred/*.png"))
        det_masks = list((base_path / "predictions").glob("*/symptoms_det/pred/*.png"))
        rgb_imgs = list((base_path / "inference_crops").glob("*/*.png"))
    else:
        seg_masks = list((base_path / "aligned").glob("*/masks/masks/*.png"))
        det_masks = list((base_path / "aligned").glob("*/masks/masks/*.png"))
        rgb_imgs = list((base_path / "aligned").glob("*/images/aligned/*.png"))

    out_path = base_path / "metrics"
    leaf_data_path = out_path / "leaf"
    lesion_data_path = out_path / "lesion"

    for p in [out_path, leaf_data_path, lesion_data_path]:
        p.mkdir(parents=True, exist_ok=True)

    main(aligned=aligned)