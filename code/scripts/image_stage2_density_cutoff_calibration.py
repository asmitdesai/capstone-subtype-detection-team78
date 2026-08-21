"""
Stage 2, density tail-trim cutoff calibration.

Follows image_stage2_confound_check.py's finding (real systematic
hematoxylin-density difference between cvPTC/fvPTC tiles, p=2.56e-6,
Cohen's d=0.70). Per the approved direction: trim both tails of the density
distribution (not the whole thing -- that would fight real biology), with
cutoffs chosen by visually confirming tile content at candidate values, not
a blind percentile cut.

This script:
1. Re-extracts the SAME 200-tile calibration set as the confound check
   (same seed, same case selection, same sampling -- fully reproducible),
   saving every tile to disk with its density score in the filename.
2. Builds small contact sheets of the tiles nearest several candidate low-
   and high-cutoff density values, for visual inspection.
3. Reports, for each candidate cutoff, the per-class tile-count impact on
   this calibration set -- flagging if the low cutoff disproportionately
   shrinks fvPTC (the smaller class).
"""
import numpy as np
from pathlib import Path
from PIL import Image
from wsidicom import WsiDicom
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu

SLIDES_ROOT = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides")
OUT_DIR = Path(r"D:\Capstone_Team78\code\thyroid\stage2_confound_check")
TILES_DIR = OUT_DIR / 'calibration_tiles'
TILES_DIR.mkdir(parents=True, exist_ok=True)

N_CASES_PER_CLASS = 5
N_TILES_PER_CASE = 20
NATIVE_TILE_SIZE = 1024
FINAL_TILE_SIZE = 512
MEAN_ACCEPT_THRESHOLD = 220
SEED = 42  # identical to image_stage2_confound_check.py -- same tile set

CANDIDATE_LOW_CUTOFFS = [0.02, 0.04, 0.06, 0.08]
CANDIDATE_HIGH_CUTOFFS = [0.30, 0.35, 0.40, 0.45]


def find_study_dir(case_dir):
    subdirs = [d for d in case_dir.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else case_dir


def sample_tiles_for_case(case_dir, n_tiles, rng):
    study_dir = find_study_dir(case_dir)
    slide = WsiDicom.open(str(study_dir))
    base_level = slide.levels.base_level
    thumbnail = slide.read_thumbnail((1024, 1024))
    thumb_arr = np.array(thumbnail.convert('RGB'))
    gray = thumb_arr.mean(axis=2)
    tissue_mask = gray < 200
    ys, xs = np.where(tissue_mask)
    if len(xs) == 0:
        slide.close()
        return []

    scale_x = base_level.size.width / thumbnail.size[0]
    scale_y = base_level.size.height / thumbnail.size[1]

    tiles = []
    attempts = 0
    max_attempts = n_tiles * 15
    seen = set()
    while len(tiles) < n_tiles and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        base_cx, base_cy = int(cx * scale_x), int(cy * scale_y)
        x = max(0, min(base_level.size.width - NATIVE_TILE_SIZE, base_cx - NATIVE_TILE_SIZE // 2))
        y = max(0, min(base_level.size.height - NATIVE_TILE_SIZE, base_cy - NATIVE_TILE_SIZE // 2))
        if (x, y) in seen:
            continue
        region = slide.read_region((x, y), base_level.level, (NATIVE_TILE_SIZE, NATIVE_TILE_SIZE)).convert('RGB')
        arr = np.array(region)
        if arr.mean() >= MEAN_ACCEPT_THRESHOLD:
            continue
        seen.add((x, y))
        region_512 = region.resize((FINAL_TILE_SIZE, FINAL_TILE_SIZE), Image.LANCZOS)
        tiles.append((x, y, region_512))
    slide.close()
    return tiles


def collect_class_tiles(class_dir, n_cases, n_tiles_per_case, rng, label):
    case_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
    chosen = rng.choice(len(case_dirs), size=n_cases, replace=False)
    chosen_cases = [case_dirs[i] for i in sorted(chosen)]
    results = []
    for case_dir in chosen_cases:
        print(f'  Sampling {case_dir.name}...')
        tiles = sample_tiles_for_case(case_dir, n_tiles_per_case, rng)
        print(f'    got {len(tiles)} tiles')
        for x, y, img in tiles:
            results.append((label, case_dir.name, x, y, img))
    return results


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    print('=== Re-extracting cvPTC calibration tiles (same seed as confound check) ===')
    cv_tiles = collect_class_tiles(SLIDES_ROOT / 'cvPTC_classical', N_CASES_PER_CLASS, N_TILES_PER_CASE, rng, 'cvPTC')
    print(f'Total cvPTC tiles: {len(cv_tiles)}')

    print('\n=== Re-extracting fvPTC calibration tiles ===')
    fv_tiles = collect_class_tiles(SLIDES_ROOT / 'fvPTC_follicular_variant', N_CASES_PER_CLASS, N_TILES_PER_CASE, rng, 'fvPTC')
    print(f'Total fvPTC tiles: {len(fv_tiles)}')

    all_tiles = cv_tiles + fv_tiles

    print('\nComputing hematoxylin density for all tiles and saving to disk...')
    all_h_subsampled = []
    records = []
    for label, case_id, x, y, img in all_tiles:
        hed = rgb2hed(np.array(img))
        h = hed[:, :, 0]
        all_h_subsampled.append(h[::4, ::4].ravel())
        records.append(dict(label=label, case_id=case_id, x=x, y=y, h=h, img=img))

    pooled = np.concatenate(all_h_subsampled)
    global_threshold = threshold_otsu(pooled)
    print(f'Global Otsu threshold (should match confound check: 0.04842): {global_threshold:.5f}')

    for rec in records:
        rec['density'] = float((rec['h'] > global_threshold).mean())

    # Save every tile with density in filename.
    for rec in records:
        fname = f"{rec['label']}_{rec['case_id']}_x{rec['x']}_y{rec['y']}_d{rec['density']:.4f}.png"
        rec['path'] = TILES_DIR / fname
        rec['img'].save(rec['path'])
    print(f'Saved {len(records)} tiles to {TILES_DIR}')

    densities = np.array([r['density'] for r in records])
    print(f'\nDensity range: min={densities.min():.4f}, max={densities.max():.4f}, '
          f'p5={np.percentile(densities,5):.4f}, p95={np.percentile(densities,95):.4f}')

    def nearest_to(target, k=6):
        diffs = np.abs(densities - target)
        idx = np.argsort(diffs)[:k]
        return [records[i] for i in idx]

    def contact_sheet_for_cutoff(target, records_near, path):
        cols = len(records_near)
        sheet = Image.new('RGB', (cols * 180, 200), 'white')
        for i, rec in enumerate(records_near):
            thumb = rec['img'].resize((180, 180))
            sheet.paste(thumb, (i * 180, 0))
        sheet.save(path)
        print(f'  Saved {path.name}: ' +
              ', '.join(f"{r['label']}/d={r['density']:.4f}" for r in records_near))

    print('\n=== Candidate LOW cutoffs (near-zero = check for genuinely empty/pure-colloid, no cells) ===')
    for cutoff in CANDIDATE_LOW_CUTOFFS:
        near = nearest_to(cutoff)
        path = OUT_DIR / f'cutoff_low_{cutoff:.2f}_examples.png'
        contact_sheet_for_cutoff(cutoff, near, path)

    print('\n=== Candidate HIGH cutoffs (check for solid blood/necrosis) ===')
    for cutoff in CANDIDATE_HIGH_CUTOFFS:
        near = nearest_to(cutoff)
        path = OUT_DIR / f'cutoff_high_{cutoff:.2f}_examples.png'
        contact_sheet_for_cutoff(cutoff, near, path)

    print('\n=== Per-class tile-count impact of candidate cutoffs (on this 200-tile calibration set) ===')
    cv_dens = np.array([r['density'] for r in records if r['label'] == 'cvPTC'])
    fv_dens = np.array([r['density'] for r in records if r['label'] == 'fvPTC'])
    print(f'{"low_cutoff":>10} | {"cvPTC dropped":>14} | {"fvPTC dropped":>14} | {"cvPTC kept":>11} | {"fvPTC kept":>11}')
    for cutoff in CANDIDATE_LOW_CUTOFFS:
        cv_drop = int((cv_dens < cutoff).sum())
        fv_drop = int((fv_dens < cutoff).sum())
        print(f'{cutoff:>10.2f} | {cv_drop:>6}/{len(cv_dens):<7} | {fv_drop:>6}/{len(fv_dens):<7} | '
              f'{len(cv_dens)-cv_drop:>11} | {len(fv_dens)-fv_drop:>11}')
        if fv_drop > cv_drop * 1.5 and fv_drop > 5:
            print(f'    FLAG: low cutoff {cutoff:.2f} drops disproportionately more fvPTC tiles '
                  f'({fv_drop} vs {cv_drop}) -- fvPTC is already the smaller class.')

    print(f'\n{"high_cutoff":>10} | {"cvPTC dropped":>14} | {"fvPTC dropped":>14} | {"cvPTC kept":>11} | {"fvPTC kept":>11}')
    for cutoff in CANDIDATE_HIGH_CUTOFFS:
        cv_drop = int((cv_dens > cutoff).sum())
        fv_drop = int((fv_dens > cutoff).sum())
        print(f'{cutoff:>10.2f} | {cv_drop:>6}/{len(cv_dens):<7} | {fv_drop:>6}/{len(fv_dens):<7} | '
              f'{len(cv_dens)-cv_drop:>11} | {len(fv_dens)-fv_drop:>11}')
        if cv_drop > fv_drop * 1.5 and cv_drop > 5:
            print(f'    FLAG: high cutoff {cutoff:.2f} drops disproportionately more cvPTC tiles '
                  f'({cv_drop} vs {fv_drop}).')

    import json
    out = dict(
        global_threshold=float(global_threshold),
        density_min=float(densities.min()), density_max=float(densities.max()),
        density_p5=float(np.percentile(densities, 5)), density_p95=float(np.percentile(densities, 95)),
        candidate_low_cutoffs=CANDIDATE_LOW_CUTOFFS, candidate_high_cutoffs=CANDIDATE_HIGH_CUTOFFS,
        per_class_impact_low={str(c): dict(cv_dropped=int((cv_dens < c).sum()), fv_dropped=int((fv_dens < c).sum()))
                               for c in CANDIDATE_LOW_CUTOFFS},
        per_class_impact_high={str(c): dict(cv_dropped=int((cv_dens > c).sum()), fv_dropped=int((fv_dens > c).sum()))
                                for c in CANDIDATE_HIGH_CUTOFFS},
    )
    out_path = OUT_DIR / 'cutoff_calibration_result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved calibration result to {out_path}')
