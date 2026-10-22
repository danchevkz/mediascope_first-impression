from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .data import check_test_subset, digest, file_hash, read_manifest, write_csv, write_json
from .features import parse_ocr
from .masking_regions import SCHEMA, SYSTEM_PROMPT, USER_PROMPT, apply_masks, build_messages, mask_boxes


OCR = dict(text_detection_model_name='PP-OCRv5_server_det', text_recognition_model_name='PP-OCRv5_server_rec',
           engine='paddle_static', cpu_threads=1, enable_mkldnn=False, text_recognition_batch_size=1,
           text_det_limit_side_len=960, text_det_limit_type='max', use_doc_orientation_classify=False,
           use_doc_unwarping=False, use_textline_orientation=False)
OPTIONS = {'max_tokens': 4000, 'extra_body': {'reasoning': {'effort': 'low', 'exclude': True}}}
MODEL = 'openai/gpt-6-sol'


def parse_response(response, model):
    expected = model.removeprefix('openai/')
    actual = str(response.get('model', '')).removeprefix('openai/')
    if not re.fullmatch(re.escape(expected) + r'(?:-\d{4}-?\d{2}-?\d{2})?', actual):
        raise ValueError(f'Masking response model differs from {model}: {actual}')
    choice = response['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Incomplete masking response; inspect the retained response')
    return json.loads(choice['message']['content'])


def recognize_full_page(image, model):
    boxes = []
    starts = list(range(0, max(1, image.height - 160), 1120))
    for index, top in enumerate(starts):
        bottom = min(image.height, top + 1280)
        tile = image.crop((0, top, image.width, bottom))
        lower = 0 if index == 0 else top + 80
        upper = image.height if index == len(starts) - 1 else bottom - 80
        for result in model.predict(np.asarray(tile)[:, :, ::-1].copy()):
            for box in parse_ocr(result):
                a, b, c, d = box['bbox']
                b, d = b + top, d + top
                if lower <= (b + d) / 2 < upper:
                    boxes.append({'text': box['text'], 'confidence': box['score'],
                                  'bbox': [max(0, a), max(0, b), min(image.width, c), min(image.height, d)]})
    boxes.sort(key=lambda box: (box['bbox'][1], box['bbox'][0]))
    for i, box in enumerate(boxes, 1):
        box['id'] = f'T{i:03d}'
    return {'boxes': boxes, 'width': image.width, 'height': image.height}


def generate(args):
    from filelock import FileLock
    from openai import OpenAI
    from paddleocr import PaddleOCR
    records = read_manifest(args.manifest, args.images, expected_split='test')
    check_test_subset(records, read_manifest(args.test, expected_split='test'))
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'model': args.model, 'ocr': OCR, 'device': args.device, 'request_options': OPTIONS,
                'system': SYSTEM_PROMPT, 'user': USER_PROMPT, 'schema': SCHEMA, 'input': 'full_page',
                'inputs': [{'image': row['image'], 'sha256': file_hash(row['image_path'])} for row in records]}
    path = args.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Masking output belongs to different inputs or settings')
    write_json(path, protocol)
    with FileLock(str(args.output / 'run.lock'), timeout=0):
        detector = PaddleOCR(device=args.device, **OCR)
        with OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1',
                    max_retries=0, timeout=240) as client:
            pairs = []
            for i, row in enumerate(records):
                name = f'{i + 1:04d}.png'
                with Image.open(row['image_path']) as source:
                    image = source.convert('RGB')
                ocr_path = args.output / 'ocr' / f'{i + 1:04d}.json'
                if ocr_path.exists():
                    ocr = json.loads(ocr_path.read_text(encoding='utf-8'))
                else:
                    ocr = recognize_full_page(image, detector)
                    write_json(ocr_path, ocr)
                response_path = args.output / 'responses' / f'{i + 1:04d}.json'
                if response_path.exists():
                    response = json.loads(response_path.read_text(encoding='utf-8'))
                else:
                    response = client.chat.completions.create(model=args.model, messages=build_messages(image, ocr),
                        **OPTIONS, response_format={'type': 'json_schema', 'json_schema':
                            {'name': 'identity_masks', 'strict': True, 'schema': SCHEMA}}).model_dump(mode='json')
                    write_json(response_path, response)
                decision = parse_response(response, args.model)
                regions = mask_boxes(decision, ocr, image.width, image.height)
                masked, fraction = apply_masks(image, regions)
                for condition, pixels in (('masked', masked), ('unmasked', image)):
                    directory = args.output / condition
                    directory.mkdir(exist_ok=True)
                    pixels.save(directory / name)
                write_json(args.output / 'regions' / f'{i + 1:04d}.json',
                           {'regions': regions, 'decision': decision, 'protocol_signature': digest(protocol)})
                pairs.append({'image': row['image'], 'masked_path': f'masked/{name}', 'unmasked_path': f'unmasked/{name}',
                              'regions_path': f'regions/{i + 1:04d}.json',
                              'media_name': row.get('media_name', ''), 'url': row.get('url', ''),
                              'group_id': row.get('group_id', ''), 'source_image_sha256': file_hash(row['image_path']),
                              'true_label_factuality': row['true_label_factuality'], 'true_label_bias': row['true_label_bias'],
                              'mask_fraction': fraction, 'needs_review': decision['needs_review']})
                write_csv(args.output / 'pairs.csv', pairs)
                print(f'Masked {i + 1}/{len(records)}; human review required', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Full-page publisher-identity masking')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=Path('work/dataset/test.csv'))
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=MODEL, choices=(MODEL,))
    parser.add_argument('--device', default='cpu')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
