"""Bounded local OCR backends; no hosted upload or model download."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _vision_binary(timeout=120):
    if sys.platform != 'darwin':
        raise RuntimeError('Apple Vision OCR requires macOS')
    swift = shutil.which('swiftc')
    if not swift:
        raise RuntimeError('Apple Vision OCR requires the Swift command-line tools')
    source = Path(__file__).with_name('vision_ocr.swift')
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = Path(__file__).resolve().parents[2] / '.runtime' / 'ocr'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'vision-{fingerprint}'
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            binary = Path(temporary) / 'vision'
            subprocess.run([swift, '-O', str(source), '-o', str(binary)],
                           check=True, timeout=timeout, capture_output=True)
            os.replace(binary, target)
    return target


class LocalOCR:
    def __init__(self, backend='auto', languages='zh-Hans,en-US'):
        self.backend = backend
        self.languages = languages

    def recognize(self, data, *, suffix, page=0, timeout=60):
        deadline = time.monotonic() + timeout
        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError('Local OCR deadline exceeded')
            return value
        backend = self.backend
        if backend == 'auto':
            backend = 'vision' if sys.platform == 'darwin' else 'tesseract'
        if backend == 'disabled':
            raise RuntimeError('OCR is disabled; this page/image requires recognition')
        executable = _vision_binary(timeout=remaining()) if backend == 'vision' else shutil.which('tesseract')
        if not executable:
            raise RuntimeError('Install local Tesseract plus language data, or use Apple Vision on macOS')
        with tempfile.TemporaryDirectory(prefix='rag-ocr-') as directory:
            source = Path(directory) / ('source' + suffix)
            source.write_bytes(data)
            if backend == 'vision':
                output = subprocess.run([str(executable), str(source), str(page), self.languages],
                                        check=True, timeout=remaining(), capture_output=True)
                value = json.loads(output.stdout)
            else:
                if page:
                    renderer = shutil.which('pdftoppm')
                    if not renderer:
                        raise RuntimeError('Tesseract PDF OCR requires pdftoppm (Poppler)')
                    target = Path(directory) / 'page'
                    subprocess.run([renderer, '-f', str(page), '-l', str(page), '-singlefile',
                                    '-scale-to', '4096', '-png', str(source), str(target)],
                                   check=True, timeout=remaining(), capture_output=True)
                    source = target.with_suffix('.png')
                language_map = {'zh-Hans': 'chi_sim', 'en-US': 'eng', 'zh-Hant': 'chi_tra'}
                languages = '+'.join(language_map.get(x, x) for x in self.languages.split(','))
                output = subprocess.run([str(executable), str(source), 'stdout', '-l', languages, 'tsv'],
                                        check=True, timeout=remaining(), capture_output=True)
                value = _tesseract_result(output.stdout.decode('utf-8'))
        if not isinstance(value, dict) or not isinstance(value.get('lines'), list):
            raise ValueError('Invalid OCR output')
        return value


def _tesseract_result(tsv):
    rows = list(csv.DictReader(io.StringIO(tsv), delimiter='\t'))
    if not rows:
        raise ValueError('Empty Tesseract output')
    if len({r['page_num'] for r in rows if r['level'] == '1'}) > 1:
        raise ValueError('Multi-frame images must be converted to PDF; refusing ambiguous page locations')
    width, height = int(rows[0]['width']), int(rows[0]['height'])
    groups = {}
    for row in rows:
        if row['level'] != '5' or not row.get('text', '').strip():
            continue
        key = tuple(row[k] for k in ('page_num', 'block_num', 'par_num', 'line_num'))
        groups.setdefault(key, []).append(row)
    lines = []
    for words in groups.values():
        x, y = min(int(w['left']) for w in words), min(int(w['top']) for w in words)
        right = max(int(w['left']) + int(w['width']) for w in words)
        bottom = max(int(w['top']) + int(w['height']) for w in words)
        lines.append({'text': ' '.join(w['text'] for w in words),
                      'confidence': sum(float(w['conf']) for w in words) / (100 * len(words)),
                      'bbox': [x / width, y / height, (right-x) / width, (bottom-y) / height]})
    return {'engine': 'tesseract', 'width': width, 'height': height, 'lines': lines}
