from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


LABELS = {
    'factuality': ['LOW', 'MIXED', 'HIGH'],
    'bias': ['LEFT', 'LEFT-CENTER', 'LEAST BIASED', 'RIGHT-CENTER', 'RIGHT'],
}
IMAGE_INPUT = 'full_page_original_size'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export an empty table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def membership_signature(records):
    return digest([{key: row[key] for key in (
        'image', 'media_name', 'url', 'true_label_factuality', 'true_label_bias', 'group_id')}
        for row in records])


def read_manifest(path, image_root=None, expected_split=None):
    path = Path(path)
    if expected_split and path.stem in ('train', 'test') and path.stem != expected_split:
        raise ValueError(f'Expected the {expected_split} split, received {path.stem}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        records = list(csv.DictReader(handle))
    if any(None in row or any(value is None for value in row.values()) for row in records):
        raise ValueError('Malformed manifest rows')
    metadata_path = path.parent / 'dataset.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else None
    if metadata and path.stem in metadata.get('splits', {}):
        expected = metadata['splits'][path.stem]
        if len(records) != expected['count'] or membership_signature(records) != expected['membership_sha256']:
            raise ValueError('Prepared split membership or labels changed; prepare the dataset again')
        image_root = image_root or metadata['dataset_root']
    root = Path(image_root).resolve() if image_root is not None else None
    ids, paths, sources, urls = set(), set(), set(), set()
    for row in records:
        relative = Path(row.get('image_path', row.get('image', '')))
        if not str(relative) or str(relative) == '.':
            raise ValueError('Each row needs image_path')
        resolved = ((root or path.parent.resolve()) / relative).resolve()
        boundary = root or (path.parent.resolve() if not relative.is_absolute() else None)
        if boundary and not resolved.is_relative_to(boundary):
            raise ValueError(f'Image is outside the image root: {relative}')
        image_id = row.get('image') or relative.as_posix()
        if image_id in ids or resolved in paths:
            raise ValueError(f'Duplicate image: {image_id}')
        ids.add(image_id)
        paths.add(resolved)
        for task, allowed in LABELS.items():
            value = row.get(f'true_label_{task}', '').strip().upper()
            if value not in allowed:
                raise ValueError(f'Invalid {task} label for {image_id}: {value}')
            row[f'true_label_{task}'] = value
        for field, seen in (('media_name', sources), ('url', urls)):
            value = row.get(field, '').strip()
            if value and value in seen:
                raise ValueError(f'Duplicate source {field}: {value}')
            if value:
                seen.add(value)
        if not resolved.is_file():
            raise FileNotFoundError(f'Missing screenshot: {resolved}')
        if row.get('image_sha256') and file_hash(resolved) != row['image_sha256']:
            raise ValueError(f'Image hash mismatch: {image_id}')
        row['image'] = image_id
        row['image_path'] = str(resolved)
    if not records:
        raise ValueError('Empty manifest')
    return records


def check_disjoint(train, test):
    for field in ('image', 'image_path', 'media_name', 'url', 'group_id', 'sha256', 'image_sha256'):
        left = {row[field] for row in train if row.get(field)}
        right = {row[field] for row in test if row.get(field)}
        if left & right:
            raise ValueError(f'Train/test overlap in {field}: {sorted(left & right)[:3]}')


def check_test_subset(records, reference):
    expected = {row['image']: row for row in reference}
    for row in records:
        source = expected.get(row['image'])
        if source is None:
            raise ValueError(f'Image is not in the test split: {row["image"]}')
        if any(row[f'true_label_{task}'] != source[f'true_label_{task}'] for task in LABELS):
            raise ValueError(f'Labels differ from the test split: {row["image"]}')
        if file_hash(row['image_path']) != file_hash(source['image_path']):
            raise ValueError(f'Image differs from the test split: {row["image"]}')


def load_features(directory, records):
    import numpy as np

    directory = Path(directory)
    index = json.loads((directory / 'index.json').read_text(encoding='utf-8'))
    if index['settings'].get('input') != IMAGE_INPUT:
        raise ValueError('Features must use full-page inputs; extract them in a fresh directory')
    items = index['images']
    positions = {row['image']: i for i, row in enumerate(items)}
    if len(positions) != len(items):
        raise ValueError('Duplicate feature rows')
    arrays = {}
    for encoder in index['encoders']:
        arrays[encoder] = np.load(directory / f'{encoder}.npy', allow_pickle=False)
        if len(arrays[encoder]) != len(items) or not np.isfinite(arrays[encoder]).all():
            raise ValueError(f'Invalid {encoder} feature matrix')
    output = []
    for row in records:
        i = positions[row['image']]
        item = items[i]
        if file_hash(row['image_path']) != item['sha256']:
            raise ValueError(f'Image differs from extracted features: {row["image"]}')
        output.append({'text': item.get('text', ''), 'numeric': item.get('numeric', []),
                       **{name: values[i] for name, values in arrays.items()}})
    return output, index['settings']
