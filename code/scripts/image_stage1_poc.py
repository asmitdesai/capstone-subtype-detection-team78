"""
Image branch, Stage 1: single-slide proof of concept.

Opens ONE TCGA-THCA DICOM whole-slide image case with wsidicom (NOT
OpenSlide -- these are DICOM WSIs, OpenSlide is built for SVS/TIFF and is
the documented wrong assumption for this dataset, see
docs\HANDOVER_PROJECT_CONTEXT.md SS6), reports its pyramid structure, and
extracts ~10 tiles from a visually-tissue-looking region of the thumbnail
to confirm real tissue comes through correctly before building anything
further.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from wsidicom import WsiDicom

CASE_DIR = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides\cvPTC_classical\TCGA-BJ-A0YZ"
                 r"\1.3.6.1.4.1.5962.99.1.2996836144.501070641.1639379408688.2.0")
OUT_DIR = Path(r"D:\Capstone_Team78\code\thyroid\stage1_poc_tiles")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Final saved tile size stays 512x512 (confirmed). No native ~20x pyramid
# level exists in these DICOM files -- checked directly via pydicom on two
# cases (cvPTC TCGA-BJ-A0YZ, fvPTC TCGA-BJ-A0ZG): each case has exactly 5
# .dcm files -- base (level 0, ~40x/mpp~0.25) + 3 resampled levels at x4
# (~10x), x16 (~2.5x), x32 (~1.25x) + a thumbnail. No x2 (~20x) or x8 level
# was generated for this download batch. So: extract at native 40x with a
# 1024x1024 window (same physical field of view a 512x512 ~20x tile would
# cover), then downsample 2x in the image domain to the final 512x512 --
# avoids pyramid-level interpolation, standard practice for this situation.
NATIVE_TILE_SIZE = 1024  # extracted at level 0 (~40x)
FINAL_TILE_SIZE = 512    # after 2x image-domain downsample (~20x-equivalent)


def main():
    print(f'Opening case: {CASE_DIR}')
    slide = WsiDicom.open(str(CASE_DIR))

    print('\n=== Pyramid levels (as exposed by wsidicom) ===')
    for level in slide.levels:
        approx_mag = (10.0 / level.mpp.width) if level.mpp and level.mpp.width else None
        mag_str = f'~{approx_mag:.1f}x' if approx_mag else 'n/a'
        print(f'Level index {level.level}: size={level.size.width}x{level.size.height}, '
              f'mpp={level.mpp} ({mag_str})')
    present = sorted(lvl.level for lvl in slide.levels)
    missing = [i for i in range(min(present), max(present) + 1) if i not in present]
    print(f'Level indices present: {present}')
    print(f'Level indices missing from this pyramid (no ~2x/~8x downsample generated): {missing}')
    print('(confirmed via direct pydicom inspection: only 5 .dcm files exist per case --'
          ' base + 3 resampled levels + thumbnail -- levels 1 and 3 were simply never'
          ' generated for this TCGA download batch, not a wsidicom filtering artifact)')

    base_level = slide.levels.base_level
    print(f'\nBase level: {base_level.level}, size={base_level.size.width}x{base_level.size.height}')
    print(f'Base MPP (microns per pixel): {base_level.mpp}')
    if base_level.mpp is not None and base_level.mpp.width:
        approx_mag = 10.0 / base_level.mpp.width
        print(f'Approx base magnification: ~{approx_mag:.1f}x (10/mpp heuristic)')

    print('\n=== Overview / label availability ===')
    print('overviews:', slide.overviews)
    print('labels:', slide.labels)

    # Get a low-res thumbnail to visually pick a tissue region.
    thumb_size = (1024, 1024)
    thumbnail = slide.read_thumbnail(thumb_size)
    thumb_path = OUT_DIR / '_thumbnail.png'
    thumbnail.save(thumb_path)
    print(f'\nSaved thumbnail to {thumb_path} (size requested {thumb_size}, actual {thumbnail.size})')

    thumb_arr = np.array(thumbnail.convert('RGB'))
    # crude tissue heuristic on the thumbnail: non-white/gray pixels.
    # NOTE: do NOT use the mask centroid as the sample point -- this tissue
    # is a ring/donut shape (thyroid cross-section) with a hollow white
    # center, and the centroid of a ring lands in the empty middle. Instead
    # sample actual tissue-mask pixel coordinates directly.
    gray = thumb_arr.mean(axis=2)
    tissue_mask = gray < 200  # tissue is darker/more saturated than blank glass
    ys, xs = np.where(tissue_mask)
    if len(xs) == 0:
        raise RuntimeError('No obvious tissue region found on thumbnail.')
    print(f'Thumbnail tissue pixels: {len(xs)} / {thumb_arr.shape[0]*thumb_arr.shape[1]} '
          f'({100*len(xs)/(thumb_arr.shape[0]*thumb_arr.shape[1]):.1f}%)')

    # Map thumbnail coords -> base-level pixel coords.
    scale_x = base_level.size.width / thumbnail.size[0]
    scale_y = base_level.size.height / thumbnail.size[1]

    # Extract ~10 tiles by sampling actual tissue-mask points (with a margin
    # from the mask edge so a TILE_SIZE window at full res is more likely to
    # land fully inside tissue), verifying each at full resolution.
    rng = np.random.default_rng(42)
    n_tiles = 10
    tiles_saved = 0
    idx = 0
    attempts = 0
    max_attempts = 200
    seen_coords = set()
    while tiles_saved < n_tiles and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, len(xs))
        cx, cy = int(xs[i]), int(ys[i])
        base_cx, base_cy = int(cx * scale_x), int(cy * scale_y)
        x = max(0, min(base_level.size.width - NATIVE_TILE_SIZE, base_cx - NATIVE_TILE_SIZE // 2))
        y = max(0, min(base_level.size.height - NATIVE_TILE_SIZE, base_cy - NATIVE_TILE_SIZE // 2))
        if (x, y) in seen_coords:
            continue
        # Extract at native 40x (level 0), 1024x1024, then downsample 2x in
        # the image domain to the final 512x512 (~20x-equivalent field of
        # view) -- no ~20x pyramid level exists natively (see report above).
        region = slide.read_region((x, y), base_level.level, (NATIVE_TILE_SIZE, NATIVE_TILE_SIZE))
        region = region.convert('RGB')
        arr = np.array(region)
        mean_val = arr.mean()
        if mean_val >= 220:  # still mostly background/glass, skip
            continue
        seen_coords.add((x, y))
        region_512 = region.resize((FINAL_TILE_SIZE, FINAL_TILE_SIZE), Image.LANCZOS)
        out_path = OUT_DIR / f'tile_{idx:02d}_x{x}_y{y}_mean{mean_val:.0f}.png'
        region_512.save(out_path)
        print(f'Saved {out_path.name}  (1024->512 downsample, mean pixel value={mean_val:.1f}, lower=more tissue)')
        tiles_saved += 1
        idx += 1
    print(f'({attempts} sample attempts for {tiles_saved} accepted tiles)')

    slide.close()
    print(f'\nSaved {tiles_saved} tiles to {OUT_DIR}')

    # Build a contact sheet.
    tile_files = sorted(OUT_DIR.glob('tile_*.png'))
    if tile_files:
        thumb_tiles = [Image.open(f).resize((200, 200)) for f in tile_files]
        cols = 5
        rows = (len(thumb_tiles) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * 200, rows * 200), 'white')
        for i, t in enumerate(thumb_tiles):
            r, c = divmod(i, cols)
            sheet.paste(t, (c * 200, r * 200))
        sheet_path = OUT_DIR / '_contact_sheet.png'
        sheet.save(sheet_path)
        print(f'Saved contact sheet to {sheet_path}')


if __name__ == '__main__':
    main()
