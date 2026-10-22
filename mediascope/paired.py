from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import LABELS, file_hash, read_manifest, write_csv, write_json
from .masking_regions import MASK_COLOR, mask_bitmap


def local_path(root, value):
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Pair path is outside the dataset root')
    return result


def verify_pairs(pairs_path, images, output, reviews_path, reference=None):
    with pairs_path.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    with reviews_path.open(newline='', encoding='utf-8-sig') as handle:
        reviews = list(csv.DictReader(handle))
    if len({row['image'] for row in rows}) != len(rows) or not rows:
        raise ValueError('Expected unique, non-empty pairs')
    if len({row['image'] for row in reviews}) != len(reviews):
        raise ValueError('Duplicate review rows')
    reviewed = {row['image']: row for row in reviews}
    if set(reviewed) != {row['image'] for row in rows}:
        raise ValueError('Every candidate must have two review decisions')
    if output.exists():
        raise FileExistsError('Choose a fresh pair-verification directory')
    manifests = {'masked': [], 'unmasked': []}
    expected = {row['image']: row for row in reference} if reference is not None else None
    checks = []
    for row in rows:
        decisions = [reviewed[row['image']].get(f'reviewer_{i}', '').strip().lower() for i in (1, 2)]
        if any(value not in ('accept', 'reject') for value in decisions):
            raise ValueError('Reviews must use accept or reject')
        if 'reject' in decisions:
            continue
        for task, allowed in LABELS.items():
            if row[f'true_label_{task}'] not in allowed:
                raise ValueError('Invalid target label')
        paths = {condition: local_path(images, row[f'{condition}_path']) for condition in manifests}
        regions = json.loads(local_path(images, row['regions_path']).read_text(encoding='utf-8'))['regions']
        with Image.open(paths['unmasked']) as image:
            source = np.asarray(image.convert('RGB'))
        if expected is not None:
            original = expected.get(row['image'])
            if original is None or any(row[f'true_label_{task}'] != original[f'true_label_{task}'] for task in LABELS):
                raise ValueError('Pair identity or labels differ from the test split')
            if row.get('source_image_sha256') != file_hash(original['image_path']):
                raise ValueError('Pair source hash differs from the test split')
            with Image.open(original['image_path']) as image:
                if not np.array_equal(source, np.asarray(image.convert('RGB'))):
                    raise ValueError('Unmasked pixels differ from the test screenshot')
        with Image.open(paths['masked']) as image:
            masked = np.asarray(image.convert('RGB'))
        if source.shape != masked.shape:
            raise ValueError('Pair dimensions differ')
        selection = mask_bitmap((source.shape[1], source.shape[0]), regions)
        if not np.array_equal(source[~selection], masked[~selection]) or not (masked[selection] == MASK_COLOR).all():
            raise ValueError(f'Unexpected changed pixels: {row["image"]}')
        for condition in manifests:
            manifests[condition].append({'image': row['image'], 'image_path': row[f'{condition}_path'],
                                        'media_name': row.get('media_name', row['image']), 'url': row.get('url', ''),
                                        'group_id': row.get('group_id', ''),
                                        'source_image_sha256': row.get('source_image_sha256', ''),
                                        'image_sha256': file_hash(paths[condition]),
                                        'true_label_factuality': row['true_label_factuality'],
                                        'true_label_bias': row['true_label_bias']})
        checks.append({'image': row['image'], 'masked_sha256': file_hash(paths['masked']),
                       'unmasked_sha256': file_hash(paths['unmasked']), 'unchanged_outside_masks': True})
    if not checks:
        raise ValueError('No pair passed both reviews')
    for condition, manifest in manifests.items():
        write_csv(output / f'{condition}.csv', manifest)
    write_json(output / 'verification.json', {'pairs': checks, 'n': len(checks), 'excluded': len(rows) - len(checks)})


def compare(masked, unmasked, output):
    def read(path):
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        result = {row['image']: row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate prediction rows')
        return result
    left, right = read(masked), read(unmasked)
    if left.keys() != right.keys() or not left:
        raise ValueError('The paired evaluations cover different images')
    if any(left[key]['true_label'] != right[key]['true_label'] for key in left):
        raise ValueError('Paired targets disagree')
    from .metrics import score
    task = json.loads((masked.parent / 'metrics.json').read_text(encoding='utf-8'))['task']
    if task != json.loads((unmasked.parent / 'metrics.json').read_text(encoding='utf-8'))['task']:
        raise ValueError('The paired evaluations use different tasks')
    metrics = {condition: score(task, [rows[key]['true_label'] for key in sorted(rows)],
                               [rows[key]['predicted_label'] for key in sorted(rows)])
               for condition, rows in (('masked', left), ('unmasked', right))}
    metrics['delta'] = {key: metrics['masked'][key] - metrics['unmasked'][key]
                        for key in ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'mae', 'mse')}
    write_json(output, metrics)


def main():
    parser = argparse.ArgumentParser(description='Pixel checks and paired masking evaluation')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--test', type=Path, default=Path('work/dataset/test.csv'))
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('compare')
    p.add_argument('--masked', type=Path, required=True)
    p.add_argument('--unmasked', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'verify':
        verify_pairs(args.pairs, args.images, args.output, args.reviews,
                     read_manifest(args.test, expected_split='test'))
    else:
        compare(args.masked, args.unmasked, args.output)


if __name__ == '__main__':
    main()
