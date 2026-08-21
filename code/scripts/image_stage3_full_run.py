"""
Stage 3: full 134-case tiling run into class-labeled folders, with
resumability and multiprocessing support built in from the start (not
bolted on after) -- essential given the run's wall-clock length and this
project's history of interrupted long runs (docs\HANDOVER_2026-07-28.md).

Resumability mechanism: a per-case manifest JSON is written to
Dataset\TCGA_Tiles\_manifests\<case_id>.json ONLY after that case's tiles
are fully saved to disk. On (re)start, any case with an existing manifest
is skipped. A case interrupted mid-save has no manifest yet, so it is
correctly reprocessed from scratch (not left in a silently-half-written
state) -- tiles are written to the case's own output folder namespace
(filenames include case_id), so a restarted case's fresh tiles simply
overwrite/coexist with any partial leftovers from an interrupted attempt.

Parallelization: cases are independent (each opens its own WsiDicom slide),
so a multiprocessing.Pool with one worker per case in flight is used,
bounded by N_WORKERS.

Fault isolation (added after the 2026-08-18 crash): pool.map() propagates
the first worker exception and kills the ENTIRE pool -- one case's read
error (an OSError deep in wsidicom's DICOM frame reader, cause unclear,
possibly a transient Windows file-handle issue on this specific file) took
down all 20 in-flight workers, not just the one case that hit it, and the
run then sat crashed/idle for ~13-22 hours before anyone noticed (manifest
count frozen at 36/134, no python process running). process_one_case() now
catches and logs any exception per-case instead of letting it propagate --
a failing case is skipped (no manifest written, so it's retried on the next
run rather than silently lost) and the other 19 workers keep going.

Usage: run with CASE_LIST overridden to a small subset first to verify
behavior (see __main__), then with the full case list once approved.
"""
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

from image_stage2_filtering import (
    NATIVE_TILE_SIZE, FINAL_TILE_SIZE, THUMB_TISSUE_FRACTION_MIN,
    FULLRES_TISSUE_FRACTION_MIN, BLUR_VARIANCE_MIN, INK_PIXEL_FRACTION_MAX,
    LINE_SPAN_FRACTION_MIN, DENSITY_LOW_CUTOFF, DENSITY_HIGH_CEILING,
    find_study_dir, tissue_fraction, blur_variance, ink_fraction,
    line_span_fraction, hematoxylin_density,
)

SLIDES_ROOT = Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/Dataset/TCGA_THCA_Slides")
OUT_ROOT = Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/Dataset/TCGA_Tiles")
MANIFEST_DIR = OUT_ROOT / '_manifests'
ERROR_LOG = OUT_ROOT / '_errors.jsonl'
N_WORKERS = 20  # measured decision: 32 logical cores / 128GB RAM available;
                # memory is a non-issue (~300-400MB/worker measured), 20
                # leaves 12 cores headroom for OS + disk-I/O contention,
                # see docs\HANDOVER_PROJECT_CONTEXT.md SS6 for reasoning


def case_label(case_dir):
    return 'cvPTC' if case_dir.parent.name == 'cvPTC_classical' else 'fvPTC'


def process_one_case(case_dir):
    """Fault-isolating wrapper: any exception from _process_one_case_inner is
    caught, logged to ERROR_LOG, and returned as a status='error' result --
    it does NOT propagate to pool.map(), so one bad case can no longer take
    down the other 19 in-flight workers (see module docstring, 2026-08-18
    crash). No manifest is written for a failed case, so it's naturally
    retried on the next run rather than silently skipped forever."""
    case_id = case_dir.name
    try:
        return _process_one_case_inner(case_dir)
    except Exception as e:
        tb = traceback.format_exc()
        print(f'[{time.strftime("%H:%M:%S")}] ERROR {case_id}: {e!r} -- see {ERROR_LOG}', flush=True)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(case_id=case_id, error=repr(e), traceback=tb,
                                     time=time.strftime('%Y-%m-%d %H:%M:%S'))) + '\n')
        return dict(case_id=case_id, status='error', error=repr(e))


def _process_one_case_inner(case_dir):
    case_id = case_dir.name
    manifest_path = MANIFEST_DIR / f'{case_id}.json'
    if manifest_path.exists():
        print(f'[{time.strftime("%H:%M:%S")}] SKIP  {case_id} (already done)', flush=True)
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

    n_done = len(list(MANIFEST_DIR.glob('*.json')))
    print(f'[{time.strftime("%H:%M:%S")}] DONE  {case_id} ({label}): '
          f'final_kept={counts["final_kept"]}, elapsed={elapsed:.0f}s -- '
          f'{n_done}/134 manifests total', flush=True)

    return dict(status='processed', **manifest)


SPLIT_CSV = Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/Dataset/Datasets/image_branch_case_split.csv")


def load_full_case_list():
    import csv
    cases = []
    with open(SPLIT_CSV) as f:
        for row in csv.DictReader(f):
            label_dir = 'cvPTC_classical' if row['label'] == 'cvPTC' else 'fvPTC_follicular_variant'
            cases.append(SLIDES_ROOT / label_dir / row['case_id'])
    return cases


if __name__ == '__main__':
    # Full 134-case run, using the case-level split already built and
    # verified (Dataset\Datasets\image_branch_case_split.csv). Resumability
    # verified on a 2-case demo before this (see module docstring); the
    # manifest check/write logic is unchanged since that verification.
    case_list = load_full_case_list()
    assert len(case_list) == 134, f'expected 134 cases from split CSV, got {len(case_list)}'

    print(f'Processing {len(case_list)} case(s) with {N_WORKERS} workers...')
    print(f'Manifest dir: {MANIFEST_DIR}')
    t0 = time.time()
    with Pool(processes=min(N_WORKERS, len(case_list))) as pool:
        results = pool.map(process_one_case, case_list)
    t_wall = time.time() - t0

    n_skipped = sum(1 for r in results if r['status'] == 'skipped_already_done')
    n_processed = sum(1 for r in results if r['status'] == 'processed')
    n_errored = sum(1 for r in results if r['status'] == 'error')
    print(f'\nDone in {t_wall:.1f}s. Processed: {n_processed}, Skipped (already done): {n_skipped}, '
          f'Errored (will retry next run): {n_errored}')
    for r in results:
        if r['status'] == 'processed':
            print(f"  {r['case_id']}: final_kept={r['final_kept']}, elapsed={r['elapsed_s']:.1f}s")
        elif r['status'] == 'error':
            print(f"  {r['case_id']}: ERROR -- {r['error']} (see {ERROR_LOG})")
        else:
            print(f"  {r['case_id']}: {r['status']}")
