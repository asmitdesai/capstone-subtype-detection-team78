"""
Diagnostic: identify thin dark diagonal lines visible in some Stage 2 kept
tiles (spotted by visual inspection of stage2_kept_contact_sheet.png) that
survived the area-fraction ink filter. Reproduces the EXACT same 50-tile
sample (same seed=7, same deterministic grid-scan + filtering order as
image_stage2_filtering.py) and reports each sampled tile's ink_frac plus a
new candidate "elongation" metric (connected-component analysis on the same
dark+desaturated ink mask: does any single component span a large fraction
of the tile's width/height despite low total area?).

Does NOT change the production filter yet -- this is the evidence-gathering
step requested before deciding on a fix.
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from wsidicom import WsiDicom
from skimage.color import rgb2hed, rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import laplace

from image_stage2_filtering import (
    CASE_DIRS, NATIVE_TILE_SIZE, FINAL_TILE_SIZE, THUMB_TISSUE_FRACTION_MIN,
    FULLRES_TISSUE_FRACTION_MIN, BLUR_VARIANCE_MIN, INK_VALUE_MAX, INK_SAT_MAX,
    INK_PIXEL_FRACTION_MAX, GLOBAL_H_THRESHOLD, DENSITY_LOW_CUTOFF, DENSITY_HIGH_CEILING,
    find_study_dir, tissue_fraction, blur_variance, ink_fraction, hematoxylin_density,
)

OUT_DIR = Path(r"D:\Capstone_Team78\code\thyroid\stage2_filtering_run")


def ink_mask_of(arr_rgb):
    hsv = rgb2hsv(arr_rgb)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    return (val < INK_VALUE_MAX) & (sat < INK_SAT_MAX)


def elongation_score(mask):
    """Max fraction of tile width/height spanned by any single connected
    dark+desaturated component's bounding box, plus that component's own
    area fraction and aspect ratio. Returns dict with the worst offender."""
    h, w = mask.shape
    lab = label(mask)
    best = dict(span_frac=0.0, area_frac=0.0, aspect_ratio=0.0, bbox=None)
    for region in regionprops(lab):
        if region.area < 15:  # ignore tiny noise specks
            continue
        minr, minc, maxr, maxc = region.bbox
        span = max(maxr - minr, maxc - minc) / max(h, w)
        area_frac = region.area / (h * w)
        # aspect ratio of the bounding box (long side / short side)
        bh, bw = (maxr - minr), (maxc - minc)
        aspect = max(bh, bw) / max(1, min(bh, bw))
        if span > best['span_frac']:
            best = dict(span_frac=float(span), area_frac=float(area_frac),
                        aspect_ratio=float(aspect), bbox=(minr, minc, maxr, maxc))
    return best


def process_case_full(case_dir):
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

    records = []
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = col * NATIVE_TILE_SIZE
            y0 = row * NATIVE_TILE_SIZE
            tx0, ty0 = int(x0 * scale_x), int(y0 * scale_y)
            tx1, ty1 = int((x0 + NATIVE_TILE_SIZE) * scale_x) + 1, int((y0 + NATIVE_TILE_SIZE) * scale_y) + 1
            cell = thumb_tissue_mask[ty0:ty1, tx0:tx1]
            if cell.size == 0 or cell.mean() < THUMB_TISSUE_FRACTION_MIN:
                continue
            region = slide.read_region((x0, y0), base_level.level, (NATIVE_TILE_SIZE, NATIVE_TILE_SIZE)).convert('RGB')
            arr_full = np.array(region)
            tfrac = tissue_fraction(arr_full)
            if tfrac < FULLRES_TISSUE_FRACTION_MIN:
                continue
            region_512 = region.resize((FINAL_TILE_SIZE, FINAL_TILE_SIZE), Image.LANCZOS)
            arr_512 = np.array(region_512)
            bvar = blur_variance(arr_512)
            records.append(dict(x=x0, y=y0, arr=arr_512, blur_var=bvar))
    slide.close()

    kept = []
    for r in records:
        if r['blur_var'] < BLUR_VARIANCE_MIN:
            continue
        ifrac = ink_fraction(r['arr'])
        if ifrac > INK_PIXEL_FRACTION_MAX:
            continue
        dscore = hematoxylin_density(r['arr'])
        if dscore < DENSITY_LOW_CUTOFF or dscore > DENSITY_HIGH_CEILING:
            continue
        r['density'] = dscore
        r['ink_frac'] = ifrac
        r['case_id'] = case_dir.name
        kept.append(r)
    return kept


if __name__ == '__main__':
    all_kept = []
    for case_dir in CASE_DIRS:
        print(f'Reprocessing {case_dir.name} (identical logic to image_stage2_filtering.py)...')
        kept = process_case_full(case_dir)
        print(f'  {len(kept)} kept tiles')
        all_kept.extend(kept)

    rng = np.random.default_rng(7)
    n_show = min(50, len(all_kept))
    idx = rng.choice(len(all_kept), size=n_show, replace=False) if len(all_kept) > n_show else np.arange(len(all_kept))
    sample = [all_kept[i] for i in idx]
    print(f'Reproduced identical 50-tile sample (same seed=7): {len(sample)} tiles')

    # Compute elongation metric for each sampled tile.
    rows_report = []
    for i, r in enumerate(sample):
        mask = ink_mask_of(r['arr'])
        el = elongation_score(mask)
        rows_report.append(dict(sheet_index=i, case_id=r['case_id'], x=r['x'], y=r['y'],
                                 ink_frac=r['ink_frac'], density=r['density'],
                                 elong_span_frac=el['span_frac'], elong_area_frac=el['area_frac'],
                                 elong_aspect_ratio=el['aspect_ratio']))

    # Build a labeled contact sheet (index number overlaid on each tile).
    cols = 10
    rows = (len(sample) + cols - 1) // cols
    thumb_sz = 140
    sheet = Image.new('RGB', (cols * thumb_sz, rows * thumb_sz), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(sample):
        im = Image.fromarray(r['arr']).resize((thumb_sz, thumb_sz))
        row, col = divmod(i, cols)
        sheet.paste(im, (col * thumb_sz, row * thumb_sz))
        draw.rectangle([col * thumb_sz, row * thumb_sz, col * thumb_sz + 28, row * thumb_sz + 16], fill='yellow')
        draw.text((col * thumb_sz + 2, row * thumb_sz + 2), str(i), fill='black')
    labeled_path = OUT_DIR / 'stage2_kept_contact_sheet_LABELED.png'
    sheet.save(labeled_path)
    print(f'Saved labeled contact sheet to {labeled_path}')

    print(f'\n{"idx":>4} {"case_id":<16} {"x":>7} {"y":>7} {"ink_frac":>9} {"density":>8} '
          f'{"elong_span":>10} {"elong_area":>10} {"aspect":>7}')
    for row in rows_report:
        print(f"{row['sheet_index']:>4} {row['case_id']:<16} {row['x']:>7} {row['y']:>7} "
              f"{row['ink_frac']:>9.4f} {row['density']:>8.4f} "
              f"{row['elong_span_frac']:>10.4f} {row['elong_area_frac']:>10.5f} {row['elong_aspect_ratio']:>7.2f}")

    import json
    with open(OUT_DIR / 'ink_line_diagnostic_result.json', 'w') as f:
        json.dump(rows_report, f, indent=2)
    print(f'\nSaved full report to {OUT_DIR / "ink_line_diagnostic_result.json"}')
