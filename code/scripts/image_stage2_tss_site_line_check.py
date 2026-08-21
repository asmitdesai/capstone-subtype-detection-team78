"""
Does thin-ink-line prevalence (the new shape-based secondary filter) vary
by TCGA Tissue Source Site (TSS)? Follow-up to the TSS-site/class confound
finding (TCGA-EM-* = 54% of fvPTC cases, 0% of cvPTC cases) -- if thin-ink
contamination itself correlates with site, that's supporting evidence for
how careful the project needs to be about site effects in general.

Uses random tissue-point sampling (not full grid, for speed) across a
site-diverse case selection: EM (fvPTC-only site), plus BJ, DJ (both
classes present at these sites, so class and site can be told apart), plus
CE (cvPTC-only site) for a non-EM single-class comparison.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from wsidicom import WsiDicom
from skimage.color import rgb2hsv

from image_stage2_filtering import (
    NATIVE_TILE_SIZE, FINAL_TILE_SIZE,
    tissue_fraction, FULLRES_TISSUE_FRACTION_MIN, blur_variance, BLUR_VARIANCE_MIN,
    ink_fraction, INK_PIXEL_FRACTION_MAX, line_span_fraction, LINE_SPAN_FRACTION_MIN,
    hematoxylin_density, DENSITY_LOW_CUTOFF, DENSITY_HIGH_CEILING, find_study_dir,
)

SLIDES_ROOT = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides")
OUT_DIR = Path(r"D:\Capstone_Team78\code\thyroid\stage2_filtering_run")
N_TILES_PER_CASE = 100
SEED = 55
MEAN_ACCEPT_THRESHOLD = 220  # coarse tissue-vs-background pre-filter for random sampling


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
        tiles.append((x, y, np.array(region_512)))
    slide.close()
    return tiles


def site_of(case_id):
    parts = case_id.split('-')
    return parts[1] if len(parts) > 1 else 'UNK'


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    cv_root = SLIDES_ROOT / 'cvPTC_classical'
    fv_root = SLIDES_ROOT / 'fvPTC_follicular_variant'

    def cases_with_site(root, site, n):
        cands = sorted([d for d in root.iterdir() if d.is_dir() and site_of(d.name) == site])
        if not cands:
            return []
        idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
        return [cands[i] for i in idx]

    selection = []
    selection += [(c, 'cvPTC') for c in cases_with_site(cv_root, 'BJ', 2)]
    selection += [(c, 'fvPTC') for c in cases_with_site(fv_root, 'BJ', 2)]
    selection += [(c, 'cvPTC') for c in cases_with_site(cv_root, 'DJ', 2)]
    selection += [(c, 'fvPTC') for c in cases_with_site(fv_root, 'DJ', 2)]
    selection += [(c, 'cvPTC') for c in cases_with_site(cv_root, 'CE', 2)]
    selection += [(c, 'fvPTC') for c in cases_with_site(fv_root, 'EM', 4)]

    print(f'Case selection ({len(selection)} cases):')
    for c, cls in selection:
        print(f'  {c.name} ({site_of(c.name)}, {cls})')

    rows = []
    for case_dir, cls in selection:
        print(f'\nSampling {case_dir.name}...')
        tiles = sample_tiles_for_case(case_dir, N_TILES_PER_CASE, rng)
        print(f'  got {len(tiles)} raw tissue tiles')
        n_tissue_kept = n_blur_kept = n_ink_kept = n_line_flagged = n_final = 0
        for x, y, arr in tiles:
            tfrac = tissue_fraction(arr)
            if tfrac < FULLRES_TISSUE_FRACTION_MIN:
                continue
            n_tissue_kept += 1
            if blur_variance(arr) < BLUR_VARIANCE_MIN:
                continue
            n_blur_kept += 1
            if ink_fraction(arr) > INK_PIXEL_FRACTION_MAX:
                continue
            n_ink_kept += 1
            span_frac, area_frac, aspect = line_span_fraction(arr)
            if span_frac >= LINE_SPAN_FRACTION_MIN:
                n_line_flagged += 1
                continue
            dscore = hematoxylin_density(arr)
            if dscore < DENSITY_LOW_CUTOFF or dscore > DENSITY_HIGH_CEILING:
                continue
            n_final += 1
        rows.append(dict(case_id=case_dir.name, site=site_of(case_dir.name), cls=cls,
                          n_sampled=len(tiles), n_tissue_kept=n_tissue_kept,
                          n_reaching_line_check=n_ink_kept, n_line_flagged=n_line_flagged,
                          n_final=n_final,
                          line_flag_rate=n_line_flagged / n_ink_kept if n_ink_kept else 0.0))
        print(f'  tissue_kept={n_tissue_kept} reaching_line_check={n_ink_kept} '
              f'line_flagged={n_line_flagged} ({rows[-1]["line_flag_rate"]:.3f}) final={n_final}')

    print(f'\n{"case_id":<16} {"site":<6} {"class":<7} {"n_reach":>8} {"n_line_flag":>12} {"rate":>7}')
    for r in rows:
        print(f"{r['case_id']:<16} {r['site']:<6} {r['cls']:<7} {r['n_reaching_line_check']:>8} "
              f"{r['n_line_flagged']:>12} {r['line_flag_rate']:>7.3f}")

    em_rows = [r for r in rows if r['site'] == 'EM']
    non_em_rows = [r for r in rows if r['site'] != 'EM']
    em_total_reach = sum(r['n_reaching_line_check'] for r in em_rows)
    em_total_flag = sum(r['n_line_flagged'] for r in em_rows)
    non_em_total_reach = sum(r['n_reaching_line_check'] for r in non_em_rows)
    non_em_total_flag = sum(r['n_line_flagged'] for r in non_em_rows)
    em_rate = em_total_flag / em_total_reach if em_total_reach else 0.0
    non_em_rate = non_em_total_flag / non_em_total_reach if non_em_total_reach else 0.0
    print(f'\nEM-site aggregate line-flag rate: {em_total_flag}/{em_total_reach} = {em_rate:.4f}')
    print(f'Non-EM-site aggregate line-flag rate: {non_em_total_flag}/{non_em_total_reach} = {non_em_rate:.4f}')

    import json
    with open(OUT_DIR / 'tss_site_line_check_result.json', 'w') as f:
        json.dump(dict(rows=rows, em_rate=em_rate, non_em_rate=non_em_rate,
                        em_total_reach=em_total_reach, em_total_flag=em_total_flag,
                        non_em_total_reach=non_em_total_reach, non_em_total_flag=non_em_total_flag), f, indent=2)
    print(f'\nSaved to {OUT_DIR / "tss_site_line_check_result.json"}')
