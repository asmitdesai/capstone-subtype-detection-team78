"""
Pre-Stage-2 empirical check: is there a systematic nuclei/glandular density
difference between cvPTC- and fvPTC-labeled tiles that could let a CNN
shortcut-learn tissue composition instead of actual growth-pattern
architecture?

Method:
1. Sample 5 cvPTC cases + 5 fvPTC cases (seeded).
2. Extract ~20 tissue tiles per case using the exact Stage 1 recipe
   (1024x1024 native ~40x -> 2x image-domain downsample -> 512x512,
   tissue-mask-guided sampling with a mean-intensity accept threshold).
3. Per tile: color-deconvolve to hematoxylin channel (skimage.color.rgb2hed)
   and compute a nuclear/glandular density score = fraction of pixels above
   a threshold, with the SAME threshold applied to both classes (a global
   Otsu computed on the pooled hematoxylin-channel histogram across all
   sampled tiles from both classes) -- deliberately not a per-class or
   per-tile-normalized threshold, since that would launder away any real
   between-class difference we're trying to detect.
4. Compare the per-class density-score distributions (descriptive stats +
   Mann-Whitney U test + Cohen's d effect size) and report plainly whether
   there's a systematic difference.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from wsidicom import WsiDicom
from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from scipy.stats import mannwhitneyu

SLIDES_ROOT = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides")
OUT_DIR = Path(r"D:\Capstone_Team78\code\thyroid\stage2_confound_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CASES_PER_CLASS = 5
N_TILES_PER_CASE = 20
NATIVE_TILE_SIZE = 1024
FINAL_TILE_SIZE = 512
MEAN_ACCEPT_THRESHOLD = 220  # same as Stage 1 tissue accept rule
SEED = 42


def find_study_dir(case_dir):
    # each case dir has one study-UID subfolder containing the .dcm files
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


def collect_class_tiles(class_dir, n_cases, n_tiles_per_case, rng):
    case_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
    chosen = rng.choice(len(case_dirs), size=n_cases, replace=False)
    chosen_cases = [case_dirs[i] for i in sorted(chosen)]
    results = []  # list of (case_id, x, y, PIL image)
    for case_dir in chosen_cases:
        print(f'  Sampling {case_dir.name}...')
        tiles = sample_tiles_for_case(case_dir, n_tiles_per_case, rng)
        print(f'    got {len(tiles)} tiles')
        for x, y, img in tiles:
            results.append((case_dir.name, x, y, img))
    return results


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)

    print('=== Sampling cvPTC tiles ===')
    cv_tiles = collect_class_tiles(SLIDES_ROOT / 'cvPTC_classical', N_CASES_PER_CLASS, N_TILES_PER_CASE, rng)
    print(f'Total cvPTC tiles: {len(cv_tiles)}')

    print('\n=== Sampling fvPTC tiles ===')
    fv_tiles = collect_class_tiles(SLIDES_ROOT / 'fvPTC_follicular_variant', N_CASES_PER_CLASS, N_TILES_PER_CASE, rng)
    print(f'Total fvPTC tiles: {len(fv_tiles)}')

    # Save a few example tiles per class for visual sanity check.
    def contact_sheet(tiles, path, cols=5):
        imgs = [im.resize((150, 150)) for _, _, im in [(t[0], t[1], t[3]) for t in tiles]]
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * 150, rows * 150), 'white')
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            sheet.paste(im, (c * 150, r * 150))
        sheet.save(path)

    contact_sheet(cv_tiles[:15], OUT_DIR / 'cvPTC_sample_contact_sheet.png')
    contact_sheet(fv_tiles[:15], OUT_DIR / 'fvPTC_sample_contact_sheet.png')
    print('\nSaved sample contact sheets for visual check.')

    # --- Hematoxylin density scoring ---
    print('\nComputing hematoxylin channel for all tiles...')
    all_h = []
    cv_h_channels = []
    for _, _, _, img in cv_tiles:
        hed = rgb2hed(np.array(img))
        h = hed[:, :, 0]
        cv_h_channels.append(h)
        all_h.append(h[::4, ::4].ravel())  # subsample for global threshold calc

    fv_h_channels = []
    for _, _, _, img in fv_tiles:
        hed = rgb2hed(np.array(img))
        h = hed[:, :, 0]
        fv_h_channels.append(h)
        all_h.append(h[::4, ::4].ravel())

    pooled = np.concatenate(all_h)
    global_threshold = threshold_otsu(pooled)
    print(f'Global Otsu threshold on pooled hematoxylin channel: {global_threshold:.5f}')

    cv_density = np.array([float((h > global_threshold).mean()) for h in cv_h_channels])
    fv_density = np.array([float((h > global_threshold).mean()) for h in fv_h_channels])

    print(f'\ncvPTC density score: n={len(cv_density)}, mean={cv_density.mean():.4f}, '
          f'std={cv_density.std():.4f}, median={np.median(cv_density):.4f}')
    print(f'fvPTC density score: n={len(fv_density)}, mean={fv_density.mean():.4f}, '
          f'std={fv_density.std():.4f}, median={np.median(fv_density):.4f}')

    u_stat, p_value = mannwhitneyu(cv_density, fv_density, alternative='two-sided')
    pooled_std = np.sqrt((cv_density.std(ddof=1)**2 + fv_density.std(ddof=1)**2) / 2)
    cohens_d = (cv_density.mean() - fv_density.mean()) / pooled_std if pooled_std > 0 else float('nan')

    print(f'\nMann-Whitney U test: U={u_stat:.1f}, p={p_value:.4g}')
    print(f"Cohen's d (cvPTC - fvPTC): {cohens_d:.3f}")

    print('\n=== Verdict ===')
    if p_value < 0.05 and abs(cohens_d) > 0.3:
        verdict = (f'SYSTEMATIC DIFFERENCE DETECTED (p={p_value:.4g}, Cohen d={cohens_d:.3f}). '
                   f'This is a real confound risk (case 3b) -- nuclei/glandular density filtering '
                   f'should be built properly and applied more aggressively, not left to '
                   f'weak-supervision noise tolerance alone.')
    elif p_value < 0.05:
        verdict = (f'Statistically significant (p={p_value:.4g}) but small effect size '
                   f'(Cohen d={cohens_d:.3f}) -- a real but modest difference. Worth a moderate '
                   f'density filter as a precaution, not a full aggressive one.')
    else:
        verdict = (f'NO significant systematic difference detected (p={p_value:.4g}, '
                   f'Cohen d={cohens_d:.3f}). Distributions overlap -- shortcut-learning risk from '
                   f'tissue composition looks low (case 3a). A lenient filter dropping only the '
                   f'extreme low-content tail is proportionate.')
    print(verdict)

    import json
    out = dict(
        n_cases_per_class=N_CASES_PER_CLASS, n_tiles_per_case=N_TILES_PER_CASE,
        global_hematoxylin_threshold=float(global_threshold),
        cv_density=dict(n=len(cv_density), mean=float(cv_density.mean()), std=float(cv_density.std()),
                         median=float(np.median(cv_density)), values=cv_density.tolist()),
        fv_density=dict(n=len(fv_density), mean=float(fv_density.mean()), std=float(fv_density.std()),
                         median=float(np.median(fv_density)), values=fv_density.tolist()),
        mannwhitney_u=float(u_stat), p_value=float(p_value), cohens_d=float(cohens_d),
        verdict=verdict,
    )
    out_path = OUT_DIR / 'confound_check_result.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved full result to {out_path}')

    # Histogram comparison plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(cv_density.max(), fv_density.max()) * 1.05, 25)
    ax.hist(cv_density, bins=bins, alpha=0.6, label=f'cvPTC (n={len(cv_density)})', color='steelblue')
    ax.hist(fv_density, bins=bins, alpha=0.6, label=f'fvPTC (n={len(fv_density)})', color='indianred')
    ax.set_xlabel('Hematoxylin-channel density score (fraction of pixels above global threshold)')
    ax.set_ylabel('Tile count')
    ax.set_title(f'Nuclear/glandular density by class\nMann-Whitney p={p_value:.3g}, Cohen d={cohens_d:.3f}')
    ax.legend()
    plt.tight_layout()
    hist_path = OUT_DIR / 'density_distribution_comparison.png'
    plt.savefig(hist_path, dpi=120)
    print(f'Saved histogram comparison to {hist_path}')
