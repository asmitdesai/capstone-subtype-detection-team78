"""
Pre-Stage-3 cheap survey: estimate per-case tile-count variance and total
disk/time scaling for the full 134-case run, WITHOUT running the expensive
full-resolution filtering pipeline on every case.

Uses only the thumbnail (cheap, already-required read) + Otsu-on-saturation
tissue fraction to estimate each case's tissue area, then converts to an
estimated final-kept-tile count using the survival rate measured on the
Stage 2 verification case (3625/3668 = 98.8% of tissue-filtered tiles
survive blur/ink/density filtering).
"""
import numpy as np
from pathlib import Path
from wsidicom import WsiDicom
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

SLIDES_ROOT = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides")
NATIVE_TILE_SIZE = 1024
SURVIVAL_RATE = 3625 / 3668  # measured on the Stage 2 verification case
N_SAMPLE_CASES = 24
SEED = 123


def find_study_dir(case_dir):
    subdirs = [d for d in case_dir.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else case_dir


def survey_case(case_dir):
    study_dir = find_study_dir(case_dir)
    slide = WsiDicom.open(str(study_dir))
    base_level = slide.levels.base_level
    W, H = base_level.size.width, base_level.size.height
    thumbnail = slide.read_thumbnail((1024, 1024))
    thumb_arr = np.array(thumbnail.convert('RGB'))
    hsv = rgb2hsv(thumb_arr)
    sat = hsv[:, :, 1]
    try:
        t = threshold_otsu(sat)
        tissue_frac = float((sat > t).mean())
    except ValueError:
        tissue_frac = 0.0
    slide.close()
    total_area_px = W * H
    tissue_area_px = total_area_px * tissue_frac
    est_grid_tiles = tissue_area_px / (NATIVE_TILE_SIZE ** 2)
    est_final_tiles = est_grid_tiles * SURVIVAL_RATE
    return dict(case_id=case_dir.name, W=W, H=H, tissue_frac=tissue_frac,
                est_grid_tiles=est_grid_tiles, est_final_tiles=est_final_tiles)


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    cv_cases = sorted([d for d in (SLIDES_ROOT / 'cvPTC_classical').iterdir() if d.is_dir()])
    fv_cases = sorted([d for d in (SLIDES_ROOT / 'fvPTC_follicular_variant').iterdir() if d.is_dir()])

    n_per_class = N_SAMPLE_CASES // 2
    cv_idx = rng.choice(len(cv_cases), size=n_per_class, replace=False)
    fv_idx = rng.choice(len(fv_cases), size=n_per_class, replace=False)
    sample_cases = [cv_cases[i] for i in cv_idx] + [fv_cases[i] for i in fv_idx]

    results = []
    for case_dir in sample_cases:
        r = survey_case(case_dir)
        results.append(r)
        print(f"{r['case_id']:<16} size={r['W']}x{r['H']:<10} tissue_frac={r['tissue_frac']:.3f} "
              f"est_final_tiles={r['est_final_tiles']:.0f}")

    est_tiles = np.array([r['est_final_tiles'] for r in results])
    print(f'\n=== Estimated final-tile-count distribution across {len(results)} sampled cases ===')
    print(f'mean={est_tiles.mean():.0f}, std={est_tiles.std():.0f}, min={est_tiles.min():.0f}, '
          f'max={est_tiles.max():.0f}, median={np.median(est_tiles):.0f}')
    low_cases = [r for r in results if r['est_final_tiles'] < np.percentile(est_tiles, 10)]
    print(f'\nLowest-decile cases (possible thin/small-tissue scans, worth flagging):')
    for r in sorted(low_cases, key=lambda x: x['est_final_tiles']):
        print(f"  {r['case_id']}: est_final_tiles={r['est_final_tiles']:.0f}, tissue_frac={r['tissue_frac']:.3f}")

    total_est_tiles_134 = est_tiles.mean() * 134
    avg_tile_kb = 492  # measured average kept-tile PNG size, Stage 2 run
    per_tile_processing_s = 0.2536  # measured benchmark, full read+filter funnel
    grid_to_thumbpass_ratio = 4119 / 4119  # thumbnail prefilter ~= full grid pass count on verification case (conservative)

    est_total_disk_gb = total_est_tiles_134 * avg_tile_kb / 1024 / 1024
    # time is driven by thumbnail-prefilter-pass tiles, not just final kept tiles;
    # on the verification case, prefilter-pass (4119) / final-kept (3625) = 1.136x
    est_time_multiplier = 4119 / 3625
    est_total_time_hours = (total_est_tiles_134 * est_time_multiplier * per_tile_processing_s) / 3600

    print(f'\n=== Full 134-case scaling estimate (extrapolated from this survey + Stage 2 benchmark) ===')
    print(f'Estimated total final kept tiles: {total_est_tiles_134:.0f}')
    print(f'Estimated total disk (individual PNG tiles, ~{avg_tile_kb}KB avg): {est_total_disk_gb:.1f} GB')
    print(f'Estimated total single-threaded processing time: {est_total_time_hours:.1f} hours')

    import json
    with open(Path(r"D:\Capstone_Team78\code\thyroid\stage2_filtering_run\stage3_precheck_survey.json"), 'w') as f:
        json.dump(dict(
            per_case=results,
            summary=dict(mean=float(est_tiles.mean()), std=float(est_tiles.std()),
                         min=float(est_tiles.min()), max=float(est_tiles.max())),
            scaling_estimate_134_cases=dict(
                total_tiles=float(total_est_tiles_134),
                total_disk_gb=float(est_total_disk_gb),
                total_time_hours=float(est_total_time_hours),
            ),
        ), f, indent=2)
    print('\nSaved survey result.')
