from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import shutil

from .data import LABELS, check_disjoint, digest, file_hash, membership_signature, write_csv, write_json


def read_table(path, required):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns in {path}: {sorted(set(required) - set(reader.fieldnames or []))}')
        rows = list(reader)
    if any(None in row or any(row.get(field) is None for field in required) for row in rows):
        raise ValueError(f'Malformed CSV rows in {path}')
    return rows


def source_key(row):
    return row['media_name'].strip(), row['url'].strip()


def index_sources(rows, path):
    indexed = {}
    for row in rows:
        key = source_key(row)
        if not all(key) or key in indexed:
            raise ValueError(f'Missing or duplicate source identity in {path}: {key}')
        indexed[key] = row
    return indexed


def prepare_dataset(dataset, output):
    root, output = Path(dataset).resolve(), Path(output).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError('Write prepared manifests outside the supplied dataset')
    if output.exists():
        raise FileExistsError('Choose a fresh prepared-manifest directory')
    splits, assignments, input_hashes = {}, [], {}
    metadata_by_split = {}
    for split in ('train', 'test'):
        table_path = root / f'{split}.csv'
        metadata_path = root / split / 'metadata.csv'
        rows = read_table(table_path, ['media_name', 'url', 'true_label_factuality', 'true_label_bias'])
        metadata = read_table(metadata_path, ['file_name', 'media_name', 'url', 'fact3', 'bias5', 'group_id', 'cluster'])
        sources, by_source = index_sources(rows, table_path), index_sources(metadata, metadata_path)
        if not rows or sources.keys() != by_source.keys():
            raise ValueError(f'{split} CSV and metadata must contain exactly the same nonempty source set')
        metadata_by_split[split] = by_source
        for path in (table_path, metadata_path):
            input_hashes[path.relative_to(root).as_posix()] = file_hash(path)
        prepared, filenames = [], set()
        for row in rows:
            key = source_key(row)
            item = by_source[key]
            filename = item['file_name']
            if not filename or filename in ('.', '..') or '/' in filename or '\\' in filename or ':' in filename:
                raise ValueError(f'Unsafe screenshot filename: {filename}')
            relative = Path(split) / filename
            path = (root / relative).resolve()
            if not path.is_relative_to(root / split) or not path.is_file():
                raise FileNotFoundError(f'Missing or unsafe screenshot: {relative}')
            if filename.casefold() in filenames:
                raise ValueError(f'Duplicate screenshot filename in {split}: {filename}')
            filenames.add(filename.casefold())
            labels = {}
            for task, field in (('factuality', 'fact3'), ('bias', 'bias5')):
                label = row[f'true_label_{task}'].strip().upper()
                if label not in LABELS[task] or label != item[field].strip().upper():
                    raise ValueError(f'{split} label mismatch for {key}: {task}')
                labels[f'true_label_{task}'] = label
            group = item['group_id'].strip()
            prepared.append({**row, **labels, 'media_name': key[0], 'url': key[1],
                             'image': filename, 'image_path': str(path),
                             'image_sha256': file_hash(path),
                             'group_id': group or f'singleton:{digest(key)}',
                             'cluster': item['cluster'], 'language': item.get('language', '')})
            assignments.append({'image': filename, 'split': split, 'media_name': key[0], 'url': key[1],
                                'group_id': group, 'cluster': item['cluster'], **labels})
        splits[split] = prepared
    check_disjoint(*[[{**row, 'sha256': row['image_sha256']} for row in splits[split]] for split in ('train', 'test')])
    cluster_path = root / 'clusters_13.csv'
    clusters = read_table(cluster_path, ['cluster', 'split', 'image', 'media_name', 'url', 'factuality', 'bias'])
    clustered = set()
    for row in clusters:
        split, key = row['split'], source_key(row)
        item = metadata_by_split.get(split, {}).get(key)
        if not item or (split, key) in clustered:
            raise ValueError(f'Invalid or duplicate cluster member: {split}, {key}')
        expected = (item['file_name'], item['cluster'], item['fact3'], item['bias5'])
        actual = (row['image'], row['cluster'], row['factuality'], row['bias'])
        if actual != expected:
            raise ValueError(f'Cluster and metadata disagree for {key}')
        clustered.add((split, key))
    expected_clustered = {(split, key) for split, items in metadata_by_split.items()
                          for key, item in items.items() if item['cluster']}
    if clustered != expected_clustered:
        raise ValueError('Cluster membership is incomplete')
    input_hashes['clusters_13.csv'] = file_hash(cluster_path)
    summary = {
        'dataset_root': str(root), 'source_csv_sha256': input_hashes,
        'splits': {split: {'count': len(rows), 'membership_sha256': membership_signature(rows),
                         'labels': {task: {label: Counter(row[f'true_label_{task}'] for row in rows)[label]
                                               for label in labels} for task, labels in LABELS.items()}}
                   for split, rows in splits.items()},
    }
    output.mkdir(parents=True)
    for split, rows in splits.items():
        write_csv(output / f'{split}.csv', rows)
    write_csv(output / 'split_assignments.csv', assignments)
    shutil.copyfile(cluster_path, output / 'clusters_13.csv')
    write_json(output / 'dataset.json', summary)
    return summary
