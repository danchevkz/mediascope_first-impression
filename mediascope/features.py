from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .data import IMAGE_INPUT, file_hash, write_json


ENCODERS = {'clip': 'openai/clip-vit-base-patch32', 'siglip': 'google/siglip-base-patch16-224',
            'dinov2': 'facebook/dinov2-small'}
OCR_SETTINGS = dict(text_detection_model_name='PP-OCRv6_medium_det',
                    text_recognition_model_name='PP-OCRv6_medium_rec',
                    engine='paddle_dynamic', enable_mkldnn=False,
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, text_recognition_batch_size=1)


def parse_ocr(result):
    if hasattr(result, 'json'):
        result = result.json
        result = result() if callable(result) else result
    if isinstance(result, str):
        result = json.loads(result)
    result = result.get('res', result)
    texts, scores = result.get('rec_texts', []), result.get('rec_scores', [])
    polygons = result.get('rec_polys', result.get('dt_polys', []))
    if not len(texts) == len(scores) == len(polygons):
        raise ValueError('OCR text, score, and polygon counts differ')
    boxes = []
    for text, score, polygon in zip(texts, scores, polygons, strict=True):
        points = np.asarray(polygon)
        boxes.append({'text': str(text), 'score': float(score),
                      'bbox': [int(points[:, 0].min()), int(points[:, 1].min()),
                               int(points[:, 0].max()), int(points[:, 1].max())]})
    return boxes


def text_features(boxes):
    texts = [box['text'] for box in boxes if box['text'].strip() and box['score'] >= 0.35]
    text = re.sub(r'\s+', ' ', ' '.join(texts)).strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
    numeric = [len(words), len(set(words)) / max(1, len(words)),
               float(np.mean([box['score'] for box in boxes])) if boxes else 0.0, len(words)]
    return text, numeric


def extract(records, output, names, with_ocr=True, device='cpu', ocr_device=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {'input': IMAGE_INPUT, 'encoders': {name: ENCODERS[name] for name in names},
                'ocr': OCR_SETTINGS if with_ocr else None, 'ocr_array_order': 'RGB', 'device': device}
    if with_ocr and ocr_device and ocr_device != device:
        settings['ocr_device'] = ocr_device
    source = [{'image': row['image'], 'sha256': file_hash(row['image_path'])} for row in records]
    specification = {'settings': settings, 'images': source}
    path = output / 'extraction.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != specification:
        raise ValueError('Feature directory belongs to different images or settings')
    write_json(path, specification)
    items = []
    model = None
    for i, (row, identity) in enumerate(zip(records, source, strict=True)):
        item_path = output / 'ocr' / f'{i:06d}.json'
        if item_path.exists():
            item = json.loads(item_path.read_text(encoding='utf-8'))
            if any(item[key] != value for key, value in identity.items()):
                raise ValueError('OCR record identity mismatch')
        else:
            boxes = []
            if with_ocr:
                if model is None:
                    from paddleocr import PaddleOCR
                    model = PaddleOCR(device=ocr_device or device, **OCR_SETTINGS)
                with Image.open(row['image_path']) as image:
                    array = np.asarray(image.convert('RGB'))
                for result in model.predict(array):
                    boxes.extend(parse_ocr(result))
            text, numeric = text_features(boxes)
            item = {**identity, 'text': text, 'numeric': numeric, 'boxes': boxes}
            write_json(item_path, item)
        items.append(item)
        print(f'OCR {i + 1}/{len(records)}', flush=True)
    del model
    for name in names:
        destination = output / f'{name}.npy'
        if destination.exists():
            matrix = np.load(destination, allow_pickle=False)
            if len(matrix) != len(records) or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid completed encoder output: {name}')
            continue
        import torch
        from transformers import AutoImageProcessor, AutoModel
        torch_device = device.replace('gpu:', 'cuda:')
        processor = AutoImageProcessor.from_pretrained(ENCODERS[name], use_fast=False)
        encoder = AutoModel.from_pretrained(ENCODERS[name]).to(torch_device).eval()
        vectors = []
        for i, row in enumerate(records):
            with Image.open(row['image_path']) as image:
                image = image.convert('RGB')
            inputs = processor(images=[image], return_tensors='pt').to(torch_device)
            with torch.inference_mode():
                if name == 'dinov2':
                    vector = encoder(**inputs).last_hidden_state[:, 0, :]
                    vector = torch.nn.functional.normalize(vector, dim=1)
                else:
                    vector = encoder.get_image_features(**inputs, return_dict=True).pooler_output
                    if not torch.is_tensor(vector):
                        raise TypeError(f'Unexpected {name} image-encoder output')
            vectors.append(vector.float().cpu().numpy()[0])
            print(f'{name} {i + 1}/{len(records)}', flush=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f'Non-finite {name} embeddings')
        temporary = output / f'{name}.tmp.npy'
        np.save(temporary, matrix)
        temporary.replace(destination)
        del encoder, processor
    write_json(output / 'index.json', {'settings': settings, 'images': items, 'encoders': list(names)})
