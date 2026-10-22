from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_config import LORA_EXCLUDED_MODULES, sha256_file, write_json

SIZING = {'tile_height': 1400, 'max_tiles': 6, 'max_width': 1024}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_adapter(path, expected, train_csv=None, image_root=None):
    record = load(path / 'run_config.json')
    actual = record['settings']
    wanted = expected['settings']
    for key in ('model_id', 'task', 'quant', 'thinking', 'preprocessing', 'require_full_image', 'prompts'):
        if actual.get(key) != wanted[key]:
            raise ValueError(f'Adapter {key} differs from the training configuration: {path}')
    for key in ('target_modules', 'lora_r', 'lr', 'warmup_steps', 'num_epochs', 'batch_size',
                'grad_accum', 'seed', 'no_grad_checkpointing', 'max_steps', 'limit'):
        if actual['arguments'].get(key) != wanted['arguments'][key]:
            raise ValueError(f'Adapter training parameter {key} differs; train a matching adapter: {path}')
    if record.get('model_revision') != expected['model_revision']:
        raise ValueError(f'Adapter base-model revision differs: {path}')
    lora = load(path / 'adapter_config.json')
    wanted_lora = {'exclude_modules': LORA_EXCLUDED_MODULES, 'bias': 'none',
                   'r': 16, 'lora_alpha': 32, 'lora_dropout': 0.05}
    if any(lora.get(key) != value for key, value in wanted_lora.items()):
        raise ValueError(f'Adapter LoRA configuration differs; train a matching adapter: {path}')
    if train_csv is not None:
        from mediascope_eval import load_samples, training_inventory
        samples, _ = load_samples(train_csv, image_root, expected_split='train')
        inventory = training_inventory(samples, wanted['task'])
        fields = ('image_sha256', 'label', 'media_name', 'url', 'image', 'group_id')
        identity = lambda rows: sorted(tuple(row.get(key, '') for key in fields) for row in rows)
        if identity(record.get('training_samples', [])) != identity(inventory):
            raise ValueError('Adapter training examples differ from the supplied train split')


def command(args):
    if args.action == 'train':
        key = f'{args.model}_{args.task}'
        config = load(ROOT / 'configs/training' / f'{key}.json')
        settings = config['settings']['arguments']
        result = [sys.executable, str(ROOT / 'scripts/train_peft.py'), '--model', args.model,
                  '--task', args.task, '--csv', str(args.csv.resolve()), '--image-root', str(args.images.resolve()),
                  '--output-dir', str(args.output.resolve()), '--revision', config['model_revision']]
        for name in ('quant', 'thinking', 'target_modules', 'lora_r', 'lr', 'warmup_steps',
                     'num_epochs', 'batch_size', 'grad_accum', 'checkpoint_steps', 'save_total_limit',
                     'tile_height', 'max_tiles', 'max_width', 'seed', 'num_workers'):
            if settings.get(name) is not None:
                result += ['--' + name.replace('_', '-'), str(settings[name])]
        if settings.get('require_full_image'):
            result.append('--require-full-image')
        if settings.get('no_grad_checkpointing'):
            result.append('--no-grad-checkpointing')
        return result
    if args.action == 'local':
        settings = load(ROOT / 'configs/local_evaluation.json')[args.dataset][f'{args.model}_{args.variant}'][args.task]
        result = [sys.executable, str(ROOT / 'scripts/mediascope_eval.py'), '--predictor', 'local',
                  '--model', args.model, '--task', args.task, '--csv', str(args.csv.resolve()),
                  '--image-root', str(args.images.resolve()), '--quant', settings['quant'],
                  '--thinking', settings['thinking'], '--max-new-tokens', str(settings['generation']['max_new_tokens']),
                  '--parser', 'strict', '--out', str(args.output.resolve()), '--revision', settings['model_revision']]
        for name, value in settings['preprocessing'].items():
            result += ['--' + name.replace('_', '-'), str(value)]
        if args.variant == 'base':
            key = f'{args.model}_factuality'
            result += ['--base-only', '--processor-dir', str(args.adapters.resolve() / key)]
        else:
            key = f'{args.model}_{args.task}'
            adapter = args.adapters.resolve() / settings['adapter_key']
            if settings['adapter_key'] != key:
                raise ValueError('All evaluation conditions must use the same task-specific adapter')
            if not args.dry_run:
                validate_adapter(adapter, load(ROOT / 'configs/training' / f'{key}.json'),
                                 getattr(args, 'train_csv', Path('work/dataset/train.csv')).resolve(),
                                 getattr(args, 'train_images', Path('../data/mediascope-clustered')).resolve())
            result += ['--adapter-dir', str(adapter)]
        if args.limit:
            result += ['--limit', str(args.limit)]
        if args.resume:
            result.append('--resume')
        return result
    if args.action == 'api':
        if args.dataset == 'unmasked':
            result = [sys.executable, str(ROOT / 'scripts/eval_paired_unmasked_api.py'), '--model', args.model,
                      '--dataset-root', str(args.inputs.resolve()), '--input-dir',
                      str(args.output.resolve().parent / 'prepared_inputs_unmasked'),
                      '--output-dir', str(args.output.resolve())]
        else:
            result = [sys.executable, str(ROOT / 'scripts/eval_api.py'), '--models', args.model,
                      '--dataset-root', str(args.inputs.resolve()), '--output-dir', str(args.output.resolve()),
                      '--max-tiles', '6', '--max-width', '1024', '--tile-height', '1400',
                      '--effort', 'medium', '--max-output-tokens', '8192']
        if args.resume:
            result.append('--resume')
        return result
    raise ValueError(args.action)


def prepare(args):
    if args.output.exists():
        raise FileExistsError('Choose a fresh prepared-input directory')
    root = args.images.resolve()
    from mediascope_eval import read_manifest
    rows = read_manifest(args.csv, root, expected_split='test')
    fields = list(rows[0])
    paths, inventory = set(), []
    for row in rows:
        path = (root / row['image_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path in paths:
            raise ValueError(f'Missing, repeated, or unsafe image: {path}')
        paths.add(path)
        sha = sha256_file(path)
        if row.get('image_sha256') and row['image_sha256'] != sha:
            raise ValueError(f'Image hash mismatch: {path}')
        row['image_path'] = str(path)
        inventory.append({'image_path': str(path), 'sha256': sha})
    args.output.mkdir(parents=True)
    with (args.output / 'test.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / 'dataset_manifest.json', {
        'condition': args.dataset, 'preprocessing': SIZING, 'n': len(rows), 'images': inventory,
        'source_csv_sha256': sha256_file(args.csv), 'test_csv_sha256': sha256_file(args.output / 'test.csv')})


def main():
    parser = argparse.ArgumentParser(description='MEFI VLM experiments')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--dataset', choices=('main', 'masked', 'unmasked'), required=True)
    p.add_argument('--output', type=Path, required=True)
    for action in ('train', 'local', 'api'):
        p = sub.add_parser(action)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--dataset', choices=('main',) if action == 'train' else ('main', 'masked', 'unmasked'), default='main')
        p.add_argument('--dry-run', action='store_true')
        if action != 'api':
            p.add_argument('--model', choices=('qwen35', 'ministral'), required=True)
            p.add_argument('--task', choices=('factuality', 'bias'), required=True)
            p.add_argument('--csv', type=Path, required=True)
            p.add_argument('--images', type=Path, required=True)
        else:
            p.add_argument('--model', choices=('gpt56', 'opus5', 'kimi', 'qwen'), required=True)
            p.add_argument('--inputs', type=Path, required=True)
        if action != 'train':
            p.add_argument('--resume', action='store_true')
        if action == 'local':
            p.add_argument('--variant', choices=('base', 'finetuned'), required=True)
            p.add_argument('--adapters', type=Path, required=True)
            p.add_argument('--train-csv', type=Path, default=Path('work/dataset/train.csv'))
            p.add_argument('--train-images', type=Path, default=Path('../data/mediascope-clustered'))
            p.add_argument('--limit', type=int)
    p = sub.add_parser('prepare-api-inputs')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
        return
    if args.action == 'prepare-api-inputs':
        from eval_api import load_test_samples, save_inputs
        args.output.mkdir(parents=True, exist_ok=False)
        save_inputs(load_test_samples(args.inputs), SIZING, args.output)
        return
    if args.action == 'api' and args.dataset == 'unmasked' and not args.dry_run:
        from eval_api import load_test_samples, save_inputs
        directory = args.output.resolve().parent / 'prepared_inputs_unmasked'
        directory.mkdir(parents=True, exist_ok=True)
        save_inputs(load_test_samples(args.inputs), SIZING, directory)
    invocation = command(args)
    print(shlex.join(invocation), flush=True)
    if not args.dry_run:
        subprocess.run(invocation, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
