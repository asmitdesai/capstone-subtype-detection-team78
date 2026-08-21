"""
Stage 2: full tissue / quality / ink / density filtering pipeline, verified
on the same single test case used in Stage 1 before scaling to the full
134-case run (Stage 3).

Filtering funnel, in order:
1. Tissue detection: Otsu threshold on the HSV saturation channel of the
   slide thumbnail (glass/background is low-saturation; H&E tissue is
   strongly saturated pink/purple) -- a coarse, cheap pre-filter that
   decides which grid cells are even worth reading from the WSI at full
   resolution, then RE-VERIFIED on the full-res tile itself (same Otsu
   saturation method) since the thumbnail is a coarse approximation.
2. Blur filtering: Laplacian variance on the grayscale tile -- low variance
   means the tile lacks sharp edges (out-of-focus scan region).
3. Ink filtering: surgical margin ink (saturated blue/green/black, outside
   the pink-purple H&E hue range) -- flagged and dropped per
   docs\HANDOVER_PROJECT_CONTEXT.md SS4 (ink marks orientation, never signal).
4. Density tail-trim: hematoxylin-channel density score (same recipe as
   docs\image_branch_nuclear_density_confound_check.md), low cutoff 0.02
   (genuinely acellular tiles) and an OPTIONAL generous 0.55-0.60 ceiling
   that is an outlier safety net only, NOT confound mitigation (see that
   doc for why the high end carries no confound-mitigation signal).

Tiles: 1024x1024 native ~40x -> 2x image-domain downsample -> final 512x512,
grid-tiled (non-overlapping) across the whole slide extent -- not the
random tissue-point sampling used in Stage 1/the confound check, since
Stage 2 needs an honest systematic-tiling funnel count.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from wsidicom import WsiDicom
from skimage.color import rgb2hed, rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import laplace

CASE_DIRS = [
    Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/Dataset/TCGA_THCA_Slides/cvPTC_classical/TCGA-BJ-A0YZ"),
]
OUT_DIR = Path("/media/pesu-rf/CapstoneSSD1/Capstone_Team78/code/thyroid/stage2_filtering_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATIVE_TILE_SIZE = 1024
FINAL_TILE_SIZE = 512
THUMB_TISSUE_FRACTION_MIN = 0.10   # coarse pre-filter: worth reading at all?
FULLRES_TISSUE_FRACTION_MIN = 0.50  # re-verified tissue filter on the real tile
# Blur: a per-case 5th-percentile relative threshold was tried first and
# found (via visual check) to false-positive heavily on real sharp,
# low-texture tissue (striated muscle, sparse stroma) -- these have low
# Laplacian variance not because they're out of focus but because they lack
# fine edges, which a relative percentile cutoff can't tell apart from true
# blur. Even the case-wide minimum-variance tile among kept tissue on this
# slide was visually sharp. TCGA WSI scans are generally well-QC'd, so
# genuine focus failure appears rare/absent here. Replaced with a fixed,
# conservative absolute floor that acts as a safety net for genuinely
# pathological out-of-focus regions (should any exist elsewhere in the full
# 134-case run) rather than a routine bottom-percentile drop.
BLUR_VARIANCE_MIN = 0.003  # well below the ~0.006 observed minimum on this case
# Ink detection, calibrated empirically (see docs -- a hue-based rule was
# tried first and produced ~35% false-positive tile drops, because dense
# hematoxylin-stained nuclei clusters are dark AND fall in a similar hue
# range to ink. The actual discriminator is saturation, not hue: true ink
# (including black) is dark AND desaturated/near-achromatic (measured on a
# real ink tile: sat~0.26, std 0.27 -- noisy/undefined hue), whereas dense
# real nuclei are dark but strongly, consistently saturated purple (measured
# on a real dense-nuclei tile: sat~0.43, std 0.06). Verified against the
# confound check's single densest real-tumor tile (d=0.60): scores ~0.01%
# under this rule vs 7.6% for confirmed ink.
INK_VALUE_MAX = 0.35
INK_SAT_MAX = 0.35
INK_PIXEL_FRACTION_MAX = 0.01  # tile dropped if >1% of pixels look like ink
GLOBAL_H_THRESHOLD = 0.04842  # locked from the confound check, reused as-is
DENSITY_LOW_CUTOFF = 0.02
DENSITY_HIGH_CEILING = 0.60  # generous outlier safety net only -- see doc

# Secondary, conservative shape-based ink gate -- catches thin lines that
# survive the value+saturation rule because 2x downsample blending raises
# their measured saturation above INK_SAT_MAX. Uses darkness only (no
# saturation constraint) plus a connected-component span-fraction check, so
# it only fires on marks that are both dark AND reach across a large
# fraction of the tile -- real dense-nuclei blobs are dark but round
# (low span), real thin vessels/septae are a known, accepted false-positive
# risk here (see docs\image_branch_nuclear_density_confound_check.md).
# Reported as a SEPARATE funnel count, not merged into ink_dropped.
LINE_VALUE_MAX = 0.45
LINE_MIN_AREA_PX = 15
LINE_SPAN_FRACTION_MIN = 0.5


def find_study_dir(case_dir):
    subdirs = [d for d in case_dir.iterdir() if d.is_dir()]
    return subdirs[0] if subdirs else case_dir


def tissue_fraction(arr_rgb, sat_threshold=None):
    """Tissue pixel fraction via HSV saturation thresholding.

    IMPORTANT: `sat_threshold` should be a FIXED, slide-level threshold
    (computed once from the whole-slide thumbnail, where a genuine
    bimodal tissue-vs-background saturation distribution exists), not
    re-fit per tile. Found via DJ-A2PP spot check (2026-08-12): re-fitting
    Otsu on each individual 1024x1024 tile fails badly for tiles that are
    wholly or mostly dense tissue with no real background present -- Otsu
    still forces a bimodal split of the tile's OWN internal saturation
    variance, producing a threshold far higher than the true tissue/
    background boundary and misclassifying most of a genuinely all-tissue
    tile as "background" (measured: per-tile Otsu threshold 0.416 vs. the
    correct slide-wide 0.184 on one such tile -- tissue_fraction 0.36 vs.
    the correct 0.98). This silently rejected large fractions of real
    tissue on cases with big, dense, uniformly-stained tumor regions.
    `sat_threshold=None` falls back to per-tile Otsu for backward
    compatibility/testing only -- production code must pass the slide's
    thumbnail-derived threshold explicitly.
    """
    hsv = rgb2hsv(arr_rgb)
    sat = hsv[:, :, 1]
    if sat_threshold is not None:
        return float((sat > sat_threshold).mean())
    try:
        t = threshold_otsu(sat)
    except ValueError:
        return 0.0
    return float((sat > t).mean())


def blur_variance(arr_rgb):
    gray = rgb2gray(arr_rgb)
    return float(laplace(gray).var())


def ink_fraction(arr_rgb):
    hsv = rgb2hsv(arr_rgb)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    ink_mask = (val < INK_VALUE_MAX) & (sat < INK_SAT_MAX)
    return float(ink_mask.mean())


def line_span_fraction(arr_rgb):
    """Max span-fraction (largest dimension of any dark connected
    component's bounding box, relative to tile size) -- the secondary
    thin-line detector. Returns (span_frac, area_frac, aspect_ratio) of the
    worst-offending component."""
    hsv = rgb2hsv(arr_rgb)
    val = hsv[:, :, 2]
    mask = val < LINE_VALUE_MAX
    h, w = mask.shape
    lab = label(mask)
    best_span = 0.0
    best = (0.0, 0.0, 0.0)
    for region in regionprops(lab):
        if region.area < LINE_MIN_AREA_PX:
            continue
        minr, minc, maxr, maxc = region.bbox
        span = max(maxr - minr, maxc - minc) / max(h, w)
        if span > best_span:
            bh, bw = maxr - minr, maxc - minc
            aspect = max(bh, bw) / max(1, min(bh, bw))
            best_span = span
            best = (float(span), float(region.area / (h * w)), float(aspect))
    return best


def hematoxylin_density(arr_rgb):
    hed = rgb2hed(arr_rgb)
    h = hed[:, :, 0]
    return float((h > GLOBAL_H_THRESHOLD).mean())


def process_case(case_dir):
    study_dir = find_study_dir(case_dir)
    print(f'\n=== Processing {case_dir.name} ===')
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
    print(f'Slide size: {W}x{H}. Grid: {n_cols} x {n_rows} = {total_grid_cells} cells '
          f'(non-overlapping {NATIVE_TILE_SIZE}px)')

    records = []
    counts = dict(grid_total=total_grid_cells, thumb_prefilter_pass=0,
                  fullres_tissue_kept=0, blur_dropped=0, ink_dropped=0,
                  line_dropped=0, density_low_dropped=0, density_high_dropped=0, final_kept=0)

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

            bvar = blur_variance(arr_512)
            records.append(dict(x=x0, y=y0, arr=arr_512, blur_var=bvar, stage='blur_check'))

    slide.close()
    print(f'Grid cells: {counts["grid_total"]}, thumbnail pre-filter pass: {counts["thumb_prefilter_pass"]}, '
          f'full-res tissue kept: {counts["fullres_tissue_kept"]}')

    # Report this case's blur-variance distribution for the record, but use
    # the fixed absolute safety-net floor (see BLUR_VARIANCE_MIN comment),
    # not a per-case relative percentile.
    blur_vars = np.array([r['blur_var'] for r in records])
    print(f'Blur variance stats: min={blur_vars.min():.5f}, p5={np.percentile(blur_vars,5):.5f}, '
          f'median={np.median(blur_vars):.5f}, max={blur_vars.max():.5f}')
    blur_threshold = BLUR_VARIANCE_MIN
    print(f'Blur variance threshold (fixed absolute safety-net floor): {blur_threshold:.5f}')

    kept = []
    blur_rejects, ink_rejects, line_rejects = [], [], []
    for r in records:
        if r['blur_var'] < blur_threshold:
            counts['blur_dropped'] += 1
            if len(blur_rejects) < 20:
                blur_rejects.append(r)
            continue
        ifrac = ink_fraction(r['arr'])
        if ifrac > INK_PIXEL_FRACTION_MAX:
            counts['ink_dropped'] += 1
            r['ink_frac'] = ifrac
            if len(ink_rejects) < 20:
                ink_rejects.append(r)
            continue
        span_frac, area_frac, aspect = line_span_fraction(r['arr'])
        if span_frac >= LINE_SPAN_FRACTION_MIN:
            counts['line_dropped'] += 1
            r['ink_frac'] = ifrac
            r['line_span_frac'] = span_frac
            if len(line_rejects) < 20:
                line_rejects.append(r)
            continue
        dscore = hematoxylin_density(r['arr'])
        if dscore < DENSITY_LOW_CUTOFF:
            counts['density_low_dropped'] += 1
            continue
        if dscore > DENSITY_HIGH_CEILING:
            counts['density_high_dropped'] += 1
            continue
        r['density'] = dscore
        r['ink_frac'] = ifrac
        kept.append(r)

    # Save reject samples for visual QA.
    def save_reject_sheet(rejects, path, label_fn):
        if not rejects:
            return
        cols = min(10, len(rejects))
        rows = (len(rejects) + cols - 1) // cols
        sz = 140
        sheet = Image.new('RGB', (cols * sz, rows * sz), 'white')
        for i, r in enumerate(rejects):
            im = Image.fromarray(r['arr']).resize((sz, sz))
            row, col = divmod(i, cols)
            sheet.paste(im, (col * sz, row * sz))
        sheet.save(path)
        print(f'  Saved {len(rejects)} reject examples to {path.name}: ' +
              ', '.join(label_fn(r) for r in rejects[:5]) + ' ...')

    save_reject_sheet(blur_rejects, OUT_DIR / f'{case_dir.name}_blur_rejects.png',
                       lambda r: f"var={r['blur_var']:.4f}")
    save_reject_sheet(ink_rejects, OUT_DIR / f'{case_dir.name}_ink_rejects.png',
                       lambda r: f"ink_frac={r['ink_frac']:.3f}")
    save_reject_sheet(line_rejects, OUT_DIR / f'{case_dir.name}_line_rejects.png',
                       lambda r: f"span={r['line_span_frac']:.3f}")
    ink_reject_dir = OUT_DIR / f'{case_dir.name}_ink_reject_tiles'
    ink_reject_dir.mkdir(exist_ok=True)
    for r in ink_rejects[:20]:
        print(f"    ink reject: x={r['x']} y={r['y']} ink_frac={r['ink_frac']:.4f}")
        Image.fromarray(r['arr']).save(ink_reject_dir / f"x{r['x']}_y{r['y']}_ink{r['ink_frac']:.4f}.png")

    counts['final_kept'] = len(kept)
    print(f'\n=== Filtering funnel for {case_dir.name} ===')
    for k, v in counts.items():
        print(f'  {k}: {v}')

    return counts, kept, blur_threshold


if __name__ == '__main__':
    all_kept = []
    all_counts = {}
    for case_dir in CASE_DIRS:
        counts, kept, blur_threshold = process_case(case_dir)
        all_counts[case_dir.name] = dict(counts=counts, blur_threshold=blur_threshold)
        for k in kept:
            k['case_id'] = case_dir.name
        all_kept.extend(kept)

    print(f'\n=== TOTAL kept tiles across {len(CASE_DIRS)} case(s): {len(all_kept)} ===')

    # Save ~50 kept tiles as a contact sheet (random sample if more than 50).
    rng = np.random.default_rng(7)
    n_show = min(50, len(all_kept))
    idx = rng.choice(len(all_kept), size=n_show, replace=False) if len(all_kept) > n_show else np.arange(len(all_kept))
    sample = [all_kept[i] for i in idx]

    cols = 10
    rows = (len(sample) + cols - 1) // cols
    thumb_sz = 140
    sheet = Image.new('RGB', (cols * thumb_sz, rows * thumb_sz), 'white')
    for i, r in enumerate(sample):
        im = Image.fromarray(r['arr']).resize((thumb_sz, thumb_sz))
        row, col = divmod(i, cols)
        sheet.paste(im, (col * thumb_sz, row * thumb_sz))
    sheet_path = OUT_DIR / 'stage2_kept_contact_sheet.png'
    sheet.save(sheet_path)
    print(f'Saved contact sheet of {n_show} kept tiles to {sheet_path}')

    # Also save every kept tile individually for spot-checking.
    tiles_dir = OUT_DIR / 'kept_tiles'
    tiles_dir.mkdir(exist_ok=True)
    for r in all_kept:
        fname = f"{r['case_id']}_x{r['x']}_y{r['y']}_d{r['density']:.4f}_blur{r['blur_var']:.0f}.png"
        Image.fromarray(r['arr']).save(tiles_dir / fname)
    print(f'Saved {len(all_kept)} individual kept tiles to {tiles_dir}')

    import json
    with open(OUT_DIR / 'stage2_filtering_result.json', 'w') as f:
        json.dump(dict(
            params=dict(
                native_tile_size=NATIVE_TILE_SIZE, final_tile_size=FINAL_TILE_SIZE,
                thumb_tissue_fraction_min=THUMB_TISSUE_FRACTION_MIN,
                fullres_tissue_fraction_min=FULLRES_TISSUE_FRACTION_MIN,
                ink_value_max=INK_VALUE_MAX, ink_sat_max=INK_SAT_MAX,
                ink_pixel_fraction_max=INK_PIXEL_FRACTION_MAX,
                blur_variance_min=BLUR_VARIANCE_MIN,
                line_value_max=LINE_VALUE_MAX, line_min_area_px=LINE_MIN_AREA_PX,
                line_span_fraction_min=LINE_SPAN_FRACTION_MIN,
                global_h_threshold=GLOBAL_H_THRESHOLD,
                density_low_cutoff=DENSITY_LOW_CUTOFF, density_high_ceiling=DENSITY_HIGH_CEILING,
            ),
            per_case=all_counts,
            total_kept=len(all_kept),
        ), f, indent=2)
    print(f'Saved full result to {OUT_DIR / "stage2_filtering_result.json"}')
