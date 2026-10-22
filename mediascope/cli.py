from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold

from .classifiers import FeatureClassifier, METHODS, parameter_grid
from .data import LABELS, check_disjoint, digest, file_hash, load_features, read_manifest, write_csv, write_json
from .metrics import score


def export_predictions(output, task, method, records, predicted):
    truth = [row[f'true_label_{task}'] for row in records]
    metrics = dict(task=task, method=method, **score(task, truth, predicted))
    write_json(output / 'metrics.json', metrics)
    write_csv(output / 'predictions.csv', [
        {'image': row['image'], 'true_label': true, 'predicted_label': str(pred)}
        for row, true, pred in zip(records, truth, predicted, strict=True)])
    return metrics


def train(args):
    records = read_manifest(args.train, args.images, expected_split='train')
    features, settings = load_features(args.features, records)
    if args.output.exists():
        raise FileExistsError('Choose a fresh training output directory')
    args.output.mkdir(parents=True)
    for task in args.tasks:
        target = np.asarray([row[f'true_label_{task}'] for row in records])
        for method in args.methods:
            model = FeatureClassifier(method=method)
            directory = args.output / task / method
            directory.mkdir(parents=True)
            if args.search:
                group_flags = [bool(row.get('group_id')) for row in records]
                if any(group_flags) and not all(group_flags):
                    raise ValueError('group_id must be present for every training row or none')
                groups = np.array([row['group_id'] for row in records]) if all(group_flags) else None
                cv_class = StratifiedGroupKFold if groups is not None else StratifiedKFold
                cv = cv_class(n_splits=args.folds, shuffle=True, random_state=83)
                search = GridSearchCV(model, parameter_grid(method), cv=cv, n_jobs=args.jobs,
                                      scoring=make_scorer(f1_score, average='macro', labels=LABELS[task],
                                                          pos_label=None, zero_division=0),
                                      error_score='raise', refit=True)
                search.fit(features, target, groups=groups)
                model = search.best_estimator_
                write_json(directory / 'search.json', {
                    'best_params': search.best_params_, 'best_macro_f1': search.best_score_,
                    'folds': args.folds, 'grouped': groups is not None,
                    'candidates': [{'params': params, 'mean_macro_f1': float(mean), 'std_macro_f1': float(std)}
                                   for params, mean, std in zip(search.cv_results_['params'],
                                       search.cv_results_['mean_test_score'], search.cv_results_['std_test_score'], strict=True)]})
            else:
                model.fit(features, target)
            identity = [{'image': row['image'], 'image_path': row['image_path'],
                         'media_name': row.get('media_name', ''), 'url': row.get('url', ''),
                         'group_id': row.get('group_id', ''), 'sha256': file_hash(row['image_path'])} for row in records]
            artifact = {'model': model, 'task': task, 'method': method, 'training_rows': identity,
                        'feature_settings': settings, 'training_signature': digest(records)}
            joblib.dump(artifact, directory / 'model.joblib', compress=3)
            write_json(directory / 'parameters.json', model.get_params())
            print(f'Trained {task}: {METHODS[method]} ({len(records)} rows)', flush=True)


def predict(args):
    records = read_manifest(args.test, args.images, expected_split='test')
    features, settings = load_features(args.features, records)
    if args.output.exists():
        raise FileExistsError('Choose a fresh evaluation directory')
    artifact = joblib.load(args.model)
    for row in records:
        row['sha256'] = file_hash(row['image_path'])
    check_disjoint(artifact['training_rows'], records)
    expected = {key: value for key, value in artifact['feature_settings'].items() if key != 'device'}
    actual = {key: value for key, value in settings.items() if key != 'device'}
    if actual != expected:
        raise ValueError('Training and test feature extraction settings differ')
    predicted = artifact['model'].predict(features)
    result = export_predictions(args.output, artifact['task'], artifact['method'], records, predicted)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_class', 'confusion_matrix')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description='MEFI screenshot classification experiments')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--dataset', type=Path, default=Path('../data/mediascope-clustered'))
    p.add_argument('--output', type=Path, default=Path('work/dataset'))
    p = sub.add_parser('extract')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--encoders', nargs='*', choices=('clip', 'siglip', 'dinov2'), default=['clip', 'siglip', 'dinov2'])
    p.add_argument('--no-ocr', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--ocr-device', help='OCR device, defaults to --device')
    p = sub.add_parser('train')
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    p.add_argument('--tasks', nargs='+', choices=list(LABELS), default=list(LABELS))
    p.add_argument('--search', action='store_true')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--jobs', type=int, default=1)
    p = sub.add_parser('predict')
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        from .dataset import prepare_dataset
        print(json.dumps(prepare_dataset(args.dataset, args.output), indent=2))
    elif args.action == 'extract':
        from .features import extract
        extract(read_manifest(args.manifest, args.images), args.output, args.encoders,
                not args.no_ocr, args.device, args.ocr_device)
    elif args.action == 'train':
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
