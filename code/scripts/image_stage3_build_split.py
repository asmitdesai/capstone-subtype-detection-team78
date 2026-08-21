"""
Stage 3, step 1: build the case-level train/test split BEFORE any tiling
starts, per docs\HANDOVER_PROJECT_CONTEXT.md SS6's Stage 3 plan.

Requirements (all explicit, prior decisions):
1. Split is case-level, decided before tiling -- every tile from a case
   goes entirely into train or entirely into test (SS4).
2. The 27 downloaded cases that overlap with the 92-patient PairedDemoSet
   (the methylation branch's held-out demo cohort) are FORCED into the
   image-branch test split, so the demo can eventually run the same
   patients through both branches (image_case_ids.csv).
3. Stratify by class AND TSS site (2026-08-12 TSS-site confound decision)
   -- baseline mitigation, does NOT resolve the site confound itself (see
   docs\image_branch_tss_site_confound_check.md), only avoids compounding
   it with an additional avoidable train/test site mismatch.
4. Report tile-count balance, not just case counts -- EM cases are known
   to be severely under-tiled (668 vs 4401 mean, zero overlap with non-EM),
   so a case-count-balanced split can still leave test with very few
   actual EM tiles to evaluate the mandatory EM-vs-non-EM subgroup check on.

Tile-count estimates use the cheap thumbnail-only tissue-fraction survey
(no full-res reads) across ALL 134 cases, scaled by the REAL measured
aggregate survival rate (70.0%, post tissue_fraction-bug-fix, measured on
6 real cases -- see docs\image_branch_nuclear_density_confound_check.md).
"""
import csv
import numpy as np
from pathlib import Path
from wsidicom import WsiDicom
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

SLIDES_ROOT = Path(r"D:\Capstone_Team78\Dataset\TCGA_THCA_Slides")
DEMO_CASE_IDS_CSV = Path(r"D:\Capstone_Team78\Dataset\Datasets\PairedDemoSet\image_case_ids.csv")
OUT_CSV = Path(r"D:\Capstone_Team78\Dataset\Datasets\image_branch_case_split.csv")
OUT_DIR = Path(r"D:\Capstone_Team78\code\thyroid\stage2_filtering_run")

NATIVE_TILE_SIZE = 1024
REAL_SURVIVAL_RATE = 18548 / 26486  # measured, post-fix, on 6 real cases (2026-08-12)
TARGET_TEST_FRACTION = 0.20  # consistent with the methylation branch's 92/461 ~= 20% split
SEED = 2569  # same seed used for the methylation branch's train_test_split, for consistency


def site_of(case_id):
    parts = case_id.split('-')
    return parts[1] if len(parts) > 1 else 'UNK'


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
    grid_tiles = (W * H * tissue_frac) / (NATIVE_TILE_SIZE ** 2)
    est_final_tiles = grid_tiles * REAL_SURVIVAL_RATE
    return dict(W=W, H=H, tissue_frac=tissue_frac, est_final_tiles=est_final_tiles)


if __name__ == '__main__':
    # 1. Enumerate all 134 downloaded cases.
    cv_cases = sorted([d for d in (SLIDES_ROOT / 'cvPTC_classical').iterdir() if d.is_dir()])
    fv_cases = sorted([d for d in (SLIDES_ROOT / 'fvPTC_follicular_variant').iterdir() if d.is_dir()])
    all_cases = [(d, 'cvPTC') for d in cv_cases] + [(d, 'fvPTC') for d in fv_cases]
    print(f'Total downloaded cases: {len(all_cases)} ({len(cv_cases)} cvPTC, {len(fv_cases)} fvPTC)')

    # 2. Load PairedDemoSet overlap.
    demo_ids = set()
    with open(DEMO_CASE_IDS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            demo_ids.add(row['case_id'].strip())
    overlap = [(d, cls) for d, cls in all_cases if d.name in demo_ids]
    print(f'PairedDemoSet overlap (forced into test): {len(overlap)} cases')
    assert len(overlap) == 27, f'expected 27 overlap cases per handover, got {len(overlap)}'

    # 3. Cheap thumbnail-only survey across ALL 134 cases for tile-count estimates.
    print('\nSurveying all 134 cases (thumbnail-only, no full-res reads)...')
    survey = {}
    for i, (case_dir, cls) in enumerate(all_cases, 1):
        s = survey_case(case_dir)
        survey[case_dir.name] = s
        if i % 20 == 0:
            print(f'  {i}/{len(all_cases)}...')
    print('Survey done.')

    # 4. Build the split: force overlap into test, then stratify remaining
    # cases by (class, site) to hit ~20% overall test fraction.
    rng = np.random.default_rng(SEED)
    forced_test_ids = {d.name for d, _ in overlap}

    from collections import defaultdict
    strata = defaultdict(list)
    for case_dir, cls in all_cases:
        strata[(cls, site_of(case_dir.name))].append(case_dir)

    split = {}  # case_id -> 'train'/'test'
    for case_dir, _ in overlap:
        split[case_dir.name] = 'test'

    target_total_test = round(len(all_cases) * TARGET_TEST_FRACTION)
    n_forced = len(overlap)
    n_additional_needed = max(0, target_total_test - n_forced)
    print(f'\nTarget total test size: {target_total_test} ({TARGET_TEST_FRACTION:.0%} of {len(all_cases)})')
    print(f'Already forced into test (PairedDemoSet overlap): {n_forced}')
    print(f'Additional test cases to select via stratification: {n_additional_needed}')

    # Allocate additional test slots proportionally across strata by their
    # remaining (non-forced) case count, then randomly select within each
    # stratum (seeded).
    remaining_by_stratum = {}
    for key, cases in strata.items():
        remaining = [c for c in cases if c.name not in forced_test_ids]
        remaining_by_stratum[key] = remaining
    total_remaining = sum(len(v) for v in remaining_by_stratum.values())

    additional_test = []
    for key, remaining in remaining_by_stratum.items():
        if not remaining or total_remaining == 0:
            continue
        n_take = round(n_additional_needed * len(remaining) / total_remaining)
        n_take = min(n_take, len(remaining))
        if n_take > 0:
            idx = rng.choice(len(remaining), size=n_take, replace=False)
            additional_test.extend([remaining[i] for i in idx])

    for case_dir in additional_test:
        split[case_dir.name] = 'test'
    for case_dir, _ in all_cases:
        if case_dir.name not in split:
            split[case_dir.name] = 'train'

    # 5. Save the split CSV.
    rows = []
    for case_dir, cls in all_cases:
        s = survey[case_dir.name]
        rows.append(dict(
            case_id=case_dir.name, label=cls, tss_site=site_of(case_dir.name),
            split=split[case_dir.name],
            forced_demo_overlap=(case_dir.name in forced_test_ids),
            est_final_tiles=round(s['est_final_tiles']),
        ))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'label', 'tss_site', 'split',
                                                'forced_demo_overlap', 'est_final_tiles'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved split to {OUT_CSV} ({len(rows)} rows)')

    # 6. Report case-count balance.
    train_rows = [r for r in rows if r['split'] == 'train']
    test_rows = [r for r in rows if r['split'] == 'test']
    print(f'\n=== Case-count balance ===')
    print(f'Train: {len(train_rows)} ({sum(1 for r in train_rows if r["label"]=="cvPTC")} cvPTC, '
          f'{sum(1 for r in train_rows if r["label"]=="fvPTC")} fvPTC)')
    print(f'Test:  {len(test_rows)} ({sum(1 for r in test_rows if r["label"]=="cvPTC")} cvPTC, '
          f'{sum(1 for r in test_rows if r["label"]=="fvPTC")} fvPTC)')

    print(f'\n=== Per-site case counts (train/test) ===')
    sites = sorted(set(r['tss_site'] for r in rows))
    for site in sites:
        tr = sum(1 for r in train_rows if r['tss_site'] == site)
        te = sum(1 for r in test_rows if r['tss_site'] == site)
        print(f'  {site:<4}: train={tr:>3}  test={te:>3}')

    # 7. Report TILE-count balance -- the actual ask.
    print(f'\n=== TILE-count balance (estimated, real 70% survival rate applied) ===')
    train_tiles = sum(r['est_final_tiles'] for r in train_rows)
    test_tiles = sum(r['est_final_tiles'] for r in test_rows)
    print(f'Train: {train_tiles:,} estimated tiles')
    print(f'Test:  {test_tiles:,} estimated tiles ({100*test_tiles/(train_tiles+test_tiles):.1f}% of total)')

    print(f'\n=== TILE-count balance by EM vs non-EM ===')
    for group_name, pred in [('EM', lambda r: r['tss_site'] == 'EM'),
                              ('non-EM', lambda r: r['tss_site'] != 'EM')]:
        tr_tiles = sum(r['est_final_tiles'] for r in train_rows if pred(r))
        te_tiles = sum(r['est_final_tiles'] for r in test_rows if pred(r))
        tr_n = sum(1 for r in train_rows if pred(r))
        te_n = sum(1 for r in test_rows if pred(r))
        print(f'  {group_name:<7}: train={tr_n} cases/{tr_tiles:,} tiles   '
              f'test={te_n} cases/{te_tiles:,} tiles')

    em_test_tiles = sum(r['est_final_tiles'] for r in test_rows if r['tss_site'] == 'EM')
    em_test_cases = sum(1 for r in test_rows if r['tss_site'] == 'EM')
    fv_test_tiles = sum(r['est_final_tiles'] for r in test_rows if r['label'] == 'fvPTC')
    print(f'\nEM cases in test: {em_test_cases}, contributing {em_test_tiles:,} tiles '
          f'({100*em_test_tiles/fv_test_tiles:.1f}% of test-set fvPTC tiles, '
          f'despite being {100*em_test_cases/sum(1 for r in test_rows if r["label"]=="fvPTC"):.1f}% of test-set fvPTC cases)'
          if fv_test_tiles else '\nNo fvPTC tiles in test.')

    import json
    with open(OUT_DIR / 'stage3_split_report.json', 'w') as f:
        json.dump(dict(
            n_total=len(rows), n_train=len(train_rows), n_test=len(test_rows),
            train_tiles=train_tiles, test_tiles=test_tiles,
            em_test_cases=em_test_cases, em_test_tiles=em_test_tiles,
            rows=rows,
        ), f, indent=2)
    print(f'\nSaved full report to {OUT_DIR / "stage3_split_report.json"}')
