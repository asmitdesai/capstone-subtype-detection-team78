"""
Isolated small-scale test harness for the Linux migration -- NOT part of the
production pipeline. Mirrors image_stage3_full_run.py's per-case logic
exactly (same filters, same tile writing, same fault-isolating wrapper) but:
  - writes to a throwaway output root (never touches the real
    Dataset/TCGA_Tiles/_manifests, so it can't interfere with the 36
    already-completed cases from the interrupted Windows run)
  - takes N_WORKERS and a run-name from the command line, so the same case
    list can be re-run at different worker counts into different output
    subfolders for a clean side-by-side comparison
  - runs a small, fixed, hardcoded case list instead of the full 134-case
    split CSV

Purpose: (1) confirm wsidicom + multiprocessing behave correctly under
Linux's fork start method (Windows used spawn), (2) measure real wall-clock
throughput at different worker counts on the actual drive (a USB-attached
rotational HDD, not an SSD -- confirmed via /sys/block/sda/queue/rotational),
since the original 20-worker recommendation in docs\HANDOVER_PROJECT_CONTEXT.md
SS6 didn't have drive type as an input.

Usage: python image_stage3_smoketest.py <n_workers> <run_name> [n_cases]
"""
import sys
import json
import time
import traceback
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image
from wsidicom import WsiDicom
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).parent))
from image_stage2_filtering import (
    NATIVE_TILE_SIZE, FINAL_TILE_SIZE, THUMB_TISSUE_FRACTION_MIN,
    FULLRES_TISSUE_FRACTION_MIN, BLUR_VARIANCE_MIN, INK_PIXEL_FRACTION_MAX,
    LINE_SPAN_FRACTION_MIN, DENSITY_LOW_CUTOFF, DENSITY_HIGH_CEILING,
    find_study_dir, tissue_fraction, blur_variance, ink_fraction,
    line_span_fraction, hematoxylin_density,
)

SLIDES_ROOT = Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/Dataset/TCGA_THCA_Slides")
TEST_ROOT = Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/Dataset/_linux_migration_smoketest")

# Set from argv before Pool() is created -- forked workers inherit these at
# fork time, same pattern as image_stage3_full_run.py's module-level globals.
OUT_ROOT = None
MANIFEST_DIR = None
ERROR_LOG = None

# 20 smallest-estimated-tile-count cases NOT already in the real 36-manifest
# set (all fvPTC / TSS site EM -- also useful since the real 36 done cases
# are all cvPTC, so this incidentally smoke-tests the fvPTC path + smallest
# slides in the cohort, not just a repeat of what's already been proven).
TEST_CASE_IDS = [
    "TCGA-EM-A3O8", "TCGA-EM-A3OA", "TCGA-EM-A2CJ", "TCGA-EM-A2CT",
    "TCGA-EM-A3OB", "TCGA-EM-A3O9", "TCGA-EM-A2OW", "TCGA-EM-A3AP",
    "TCGA-EM-A3O6", "TCGA-EM-A22N", "TCGA-EM-A3FL", "TCGA-EM-A3ST",
    "TCGA-EM-A2CQ", "TCGA-EM-A3AI", "TCGA-EM-A3AL", "TCGA-EM-A2OV",
    "TCGA-EM-A3FN", "TCGA-EM-A2CP", "TCGA-EM-A2P2", "TCGA-EM-A2CO",
]


def case_label(case_dir):
    return 'cvPTC' if case_dir.parent.name == 'cvPTC_classical' else 'fvPTC'


def process_one_case(case_dir):
    try:
        return _process_one_case_inner(case_dir)
    except Exception as e:
        tb = traceback.format_exc()
        print(f'[{time.strftime("%H:%M:%S")}] ERROR {case_dir.name}: {e!r}', flush=True)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(case_id=case_dir.name, error=repr(e), traceback=tb,
                                     time=time.strftime('%Y-%m-%d %H:%M:%S'))) + '\n')
        return dict(case_id=case_dir.name, status='error', error=repr(e))


def _process_one_case_inner(case_dir):
    case_id = case_dir.name
    manifest_path = MANIFEST_DIR / f'{case_id}.json'
    if manifest_path.exists():
        return dict(case_id=case_id, status='skipped_already_done')

    label = case_label(case_dir)
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    study_dir = find_study_dir(case_dir)
    slide = WsiDicom.open(str(study_dir))
    base_level = slide.levels.base_level
    W, H = base_level.size.width, base_level.size.height

    thumbnail = slide.read_thumbnail((1024, 1024))
    thumb_arr = np.array(thumbnail.convert('RGB'))
    thumb_hsv = rgb2hsv(thumb_arr)
    thumb_sat = thumb_hsv[:, :, 1]
    thumb_t = threshold_otsu(thumb_sat)
    thumb_tissue_mask = thumb_sat > thumb_t
    scale_x = thumbnail.size[0] / W
    scale_y = thumbnail.size[1] / H

    n_cols = W // NATIVE_TILE_SIZE
    n_rows = H // NATIVE_TILE_SIZE

    counts = dict(grid_total=n_cols * n_rows, thumb_prefilter_pass=0, fullres_tissue_kept=0,
                  blur_dropped=0, ink_dropped=0, line_dropped=0,
                  density_low_dropped=0, density_high_dropped=0, final_kept=0)

    for row in range(n_rows):
        for col in range(n_cols):
            x0, y0 = col * NATIVE_TILE_SIZE, row * NATIVE_TILE_SIZE
            tx0, ty0 = int(x0 * scale_x), int(y0 * scale_y)
            tx1, ty1 = int((x0 + NATIVE_TILE_SIZE) * scale_x) + 1, int((y0 + NATIVE_TILE_SIZE) * scale_y) + 1
            cell = thumb_tissue_mask[ty0:ty1, tx0:tx1]
            if cell.size == 0 or cell.mean() < THUMB_TISSUE_FRACTION_MIN:
                continue
            counts['thumb_prefilter_pass'] += 1

            region = slide.read_region((x0, y0), base_level.level, (NATIVE_TILE_SIZE, NATIVE_TILE_SIZE)).convert('RGB')
            arr_full = np.array(region)
            if tissue_fraction(arr_full, sat_threshold=thumb_t) < FULLRES_TISSUE_FRACTION_MIN:
                continue
            counts['fullres_tissue_kept'] += 1

            region_512 = region.resize((FINAL_TILE_SIZE, FINAL_TILE_SIZE), Image.LANCZOS)
            arr_512 = np.array(region_512)

            if blur_variance(arr_512) < BLUR_VARIANCE_MIN:
                counts['blur_dropped'] += 1
                continue
            if ink_fraction(arr_512) > INK_PIXEL_FRACTION_MAX:
                counts['ink_dropped'] += 1
                continue
            span_frac, _, _ = line_span_fraction(arr_512)
            if span_frac >= LINE_SPAN_FRACTION_MIN:
                counts['line_dropped'] += 1
                continue
            dscore = hematoxylin_density(arr_512)
            if dscore < DENSITY_LOW_CUTOFF:
                counts['density_low_dropped'] += 1
                continue
            if dscore > DENSITY_HIGH_CEILING:
                counts['density_high_dropped'] += 1
                continue

            fname = f'{case_id}_x{x0}_y{y0}_d{dscore:.4f}.png'
            Image.fromarray(arr_512).save(out_dir / fname)
            counts['final_kept'] += 1

    slide.close()
    elapsed = time.time() - t0

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = dict(case_id=case_id, label=label, W=W, H=H, elapsed_s=elapsed, **counts)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'[{time.strftime("%H:%M:%S")}] DONE  {case_id} ({label}): '
          f'final_kept={counts["final_kept"]}, elapsed={elapsed:.1f}s', flush=True)

    return dict(status='processed', **manifest)


if __name__ == '__main__':
    n_workers = int(sys.argv[1])
    run_name = sys.argv[2]
    n_cases = int(sys.argv[3]) if len(sys.argv) > 3 else len(TEST_CASE_IDS)
    TEST_CASE_IDS = TEST_CASE_IDS[:n_cases]

    OUT_ROOT = TEST_ROOT / run_name
    MANIFEST_DIR = OUT_ROOT / '_manifests'
    ERROR_LOG = OUT_ROOT / '_errors.jsonl'
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    case_list = []
    for cid in TEST_CASE_IDS:
        for label_dir in ('cvPTC_classical', 'fvPTC_follicular_variant'):
            p = SLIDES_ROOT / label_dir / cid
            if p.exists():
                case_list.append(p)
                break
        else:
            raise FileNotFoundError(f'case dir not found for {cid}')

    print(f'Smoketest run "{run_name}": {len(case_list)} cases, {n_workers} workers')
    print(f'Output root: {OUT_ROOT}')
    t0 = time.time()
    with Pool(processes=min(n_workers, len(case_list))) as pool:
        results = pool.map(process_one_case, case_list)
    t_wall = time.time() - t0

    n_processed = sum(1 for r in results if r['status'] == 'processed')
    n_errored = sum(1 for r in results if r['status'] == 'error')
    total_final_kept = sum(r.get('final_kept', 0) for r in results if r['status'] == 'processed')
    total_read_tiles = sum(r.get('fullres_tissue_kept', 0) for r in results if r['status'] == 'processed')

    print(f'\n=== Run "{run_name}" ({n_workers} workers) summary ===')
    print(f'Wall time: {t_wall:.1f}s')
    print(f'Processed: {n_processed}, Errored: {n_errored}')
    print(f'Total tiles kept: {total_final_kept}, total tiles read (fullres_tissue_kept): {total_read_tiles}')
    if total_read_tiles:
        print(f'Effective read-tile throughput: {total_read_tiles / t_wall:.2f} tiles/s wall-clock')
    for r in results:
        if r['status'] == 'error':
            print(f"  {r['case_id']}: ERROR -- {r['error']}")
