"""
Real (not thumbnail-estimated) per-case final-kept-tile-count survey.
Runs the FULL Stage 2 filtering funnel (tissue/blur/ink/line/density) on a
size-diverse sample of cases, but does NOT save individual tile PNGs to
disk -- counts only, to keep this fast and memory-light. Answers: how much
does final kept-tile count actually vary case to case? Directly informs
whether a per-case tile cap is needed for Stage 3.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from wsidicom import WsiDicom
from skimage.color import rgb2hsv

from image_stage2_filtering import (
    NATIVE_TILE_SIZE, FINAL_TILE_SIZE, THUMB_TISSUE_FRACTION_MIN,
    FULLRES_TISSUE_FRACTION_MIN, BLUR_VARIANCE_MIN, INK_PIXEL_FRACTION_MAX,
    LINE_SPAN_FRACTION_MIN, DENSITY_LOW_CUTOFF, DENSITY_HIGH_CEILING,
    find_study_dir, tissue_fraction, blur_variance, ink_fraction,
    line_span_fraction, hematoxylin_density,
)
from skimage.filters import threshold_otsu

SLIDES_ROOT = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides")

# Size-diverse sample, chosen from the earlier 24-case thumbnail survey's
# estimated range (285 to 9151 estimated tiles) plus the already-verified
# test case, to span small/mid/large slides.
SAMPLE_CASES = [
    (SLIDES_ROOT / 'fvPTC_follicular_variant' / 'TCGA-EM-A3O6', 'smallest in prior survey'),
    (SLIDES_ROOT / 'fvPTC_follicular_variant' / 'TCGA-EM-A2CQ', 'small'),
    (SLIDES_ROOT / 'cvPTC_classical' / 'TCGA-CE-A3ME', 'small-mid'),
    (SLIDES_ROOT / 'cvPTC_classical' / 'TCGA-BJ-A28V', 'mid-large'),
    (SLIDES_ROOT / 'fvPTC_follicular_variant' / 'TCGA-DJ-A2PP', 'large'),
    (SLIDES_ROOT / 'fvPTC_follicular_variant' / 'TCGA-DJ-A3VM', 'largest in prior survey'),
]


def process_case_count_only(case_dir):
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
    total_grid_cells = n_cols * n_rows

    counts = dict(grid_total=total_grid_cells, thumb_prefilter_pass=0,
                  fullres_tissue_kept=0, blur_dropped=0, ink_dropped=0,
                  line_dropped=0, density_low_dropped=0, density_high_dropped=0,
                  final_kept=0)

    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * NATIVE_TILE_SIZE
            y0 = row * NATIVE_TILE_SIZE
            tx0, ty0 = int(x0 * scale_x), int(y0 * scale_y)
            tx1, ty1 = int((x0 + NATIVE_TILE_SIZE) * scale_x) + 1, int((y0 + NATIVE_TILE_SIZE) * scale_y) + 1
            cell = thumb_tissue_mask[ty0:ty1, tx0:tx1]
            if cell.size == 0 or cell.mean() < THUMB_TISSUE_FRACTION_MIN:
                continue
            counts['thumb_prefilter_pass'] += 1

            region = slide.read_region((x0, y0), base_level.level, (NATIVE_TILE_SIZE, NATIVE_TILE_SIZE)).convert('RGB')
            arr_full = np.array(region)
            tfrac = tissue_fraction(arr_full, sat_threshold=thumb_t)
            if tfrac < FULLRES_TISSUE_FRACTION_MIN:
                continue
            counts['fullres_tissue_kept'] += 1

            region_512 = region.resize((FINAL_TILE_SIZE, FINAL_TILE_SIZE), Image.LANCZOS)
            arr_512 = np.array(region_512)

            if blur_variance(arr_512) < BLUR_VARIANCE_MIN:
                counts['blur_dropped'] += 1
                continue
            ifrac = ink_fraction(arr_512)
            if ifrac > INK_PIXEL_FRACTION_MAX:
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
            counts['final_kept'] += 1

    slide.close()
    return W, H, counts


def _worker(args):
    import time
    case_dir, note = args
    t0 = time.time()
    W, H, counts = process_case_count_only(case_dir)
    elapsed = time.time() - t0
    return dict(case_id=case_dir.name, note=note, W=W, H=H, elapsed_s=elapsed, **counts)


if __name__ == '__main__':
    import time
    from multiprocessing import Pool

    t_start = time.time()
    with Pool(processes=len(SAMPLE_CASES)) as pool:
        results = pool.map(_worker, SAMPLE_CASES)
    t_wall = time.time() - t_start

    for r in results:
        print(f"\n=== {r['case_id']} ({r['note']}) === elapsed={r['elapsed_s']:.1f}s")
        print(f"Size: {r['W']}x{r['H']}")
        for k in ['grid_total', 'thumb_prefilter_pass', 'fullres_tissue_kept', 'blur_dropped',
                  'ink_dropped', 'line_dropped', 'density_low_dropped', 'density_high_dropped', 'final_kept']:
            print(f'  {k}: {r[k]}')

    sum_serial_s = sum(r['elapsed_s'] for r in results)
    print(f'\n=== Parallel batch timing ({len(results)} cases, {len(SAMPLE_CASES)} processes) ===')
    print(f'Wall clock (parallel): {t_wall:.1f}s')
    print(f'Sum of individual times (serial equivalent): {sum_serial_s:.1f}s')
    print(f'Measured speedup: {sum_serial_s / t_wall:.2f}x')

    final_counts = np.array([r['final_kept'] for r in results])
    print(f'\n=== Real per-case final-kept-tile-count summary (n={len(results)}) ===')
    for r in results:
        print(f"  {r['case_id']:<16} ({r['note']:<24}): final_kept={r['final_kept']}")
    print(f'\nmin={final_counts.min()}, max={final_counts.max()}, median={np.median(final_counts):.0f}, '
          f'mean={final_counts.mean():.0f}, std={final_counts.std():.0f}')

    import json
    with open(Path(r"D:\Capstone_Team78\code\thyroid\stage2_filtering_run\stage3_real_tilecount_survey.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved.')
