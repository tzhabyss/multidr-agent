"""Page-aware local PDF/image parsing with explicit incomplete-page policy."""
from __future__ import annotations

import io
import time
import unicodedata
from dataclasses import dataclass

from .ocr import LocalOCR


@dataclass(frozen=True)
class ParseOptions:
    ocr_backend: str = 'auto'
    ocr_languages: str = 'zh-Hans,en-US'
    ocr_timeout_seconds: float = 60
    parse_timeout_seconds: float = 300
    max_pages: int = 200
    allow_partial: bool = False
    min_text_chars: int = 40
    image_coverage_threshold: float = 0.25


class IncompleteDocumentError(ValueError):
    pass


def _canonical(text):
    return ''.join(unicodedata.normalize('NFKC', text).casefold().split())


def _bad_text(text, minimum):
    chars = [c for c in text if not c.isspace()]
    invalid = sum(c == '\ufffd' or unicodedata.category(c) in {'Cc', 'Co'} for c in chars)
    return sum(c.isalnum() for c in chars) < minimum or invalid > max(1, len(chars) * .02)


def _native_page(page):
    images = []
    resources = page.get('/Resources', {})
    resources = resources.get_object() if hasattr(resources, 'get_object') else resources
    objects = resources.get('/XObject', {})
    objects = objects.get_object() if hasattr(objects, 'get_object') else objects
    area = max(1, float(page.mediabox.width) * float(page.mediabox.height))

    def visit(operator, operands, cm, tm):
        if operator == b'Do' and operands and operands[0] in objects:
            obj = objects[operands[0]].get_object()
            if obj.get('/Subtype') == '/Image':
                images.append(min(1, abs(cm[0] * cm[3] - cm[1] * cm[2]) / area))
        elif operator == b'INLINE IMAGE':
            images.append(min(1, abs(cm[0] * cm[3] - cm[1] * cm[2]) / area))

    text = page.extract_text(visitor_operand_before=visit) or ''
    # Image XObjects nested in forms can lack an accurate visitor transform.
    has_images = bool(images) or any(
        value.get_object().get('/Subtype') in {'/Image', '/Form'}
        for value in objects.values()
    )
    content = page.get_contents()
    has_content = content is not None and bool(content.get_data().strip())
    return text, min(1, sum(images)), has_images, has_content


def _merge_ocr(native, result):
    text, records = native.rstrip(), []
    for line in result['lines']:
        value = str(line.get('text', '')).strip()
        if not value or _canonical(value) in _canonical(text):
            continue
        if text:
            text += '\n'
        start = len(text)
        text += value
        records.append({'text': value, 'start': start, 'end': len(text),
                        'bbox': line.get('bbox'), 'confidence': line.get('confidence')})
    return text, records


def parse_pdf(data, *, options=None, ocr=None, native_only=False):
    from pypdf import PdfReader
    options = options or ParseOptions()
    ocr = ocr or LocalOCR(options.ocr_backend, options.ocr_languages)
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(''):
        raise ValueError('Encrypted PDF requires a password; cannot parse')
    if not reader.pages:
        raise ValueError('PDF has no pages')
    if len(reader.pages) > options.max_pages:
        raise ValueError(f'PDF exceeds {options.max_pages}-page parsing limit')
    deadline = time.monotonic() + options.parse_timeout_seconds
    texts, records, warnings = [], [], []
    for index, page in enumerate(reader.pages):
        info = {'page': index+1, 'method': 'native', 'ocr_lines': []}
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('Document parsing deadline exceeded')
            try:
                native, coverage, has_images, has_content = _native_page(page)
            except Exception as exc:
                native, coverage, has_images, has_content = '', 0, True, True
                info['native_error'] = f'{type(exc).__name__}: {exc}'
            info.update(native_chars=len(native), image_coverage=round(coverage, 4))
            bad = _bad_text(native, options.min_text_chars)
            # Short real text without images need not be OCRed just for length.
            corrupt = _bad_text(native, 0)
            suspect = (bad and (has_images or not native.strip())) or corrupt or '\ufffd' in native
            needs_ocr = suspect or coverage >= options.image_coverage_threshold
            if not native.strip() and not has_images and not has_content:
                texts.append('')
                info['method'] = 'blank'
                records.append(info)
                continue
            if needs_ocr:
                if native_only:
                    raise IncompleteDocumentError('Page needs OCR but parser=native')
                remaining = min(options.ocr_timeout_seconds, deadline-time.monotonic())
                if remaining <= 0:
                    raise TimeoutError('Document parsing deadline exceeded')
                result = ocr.recognize(data, suffix='.pdf', page=index+1, timeout=remaining)
                # Replace a corrupt layer; otherwise merge to preserve exact digital text.
                base = '' if corrupt else native
                text, lines = _merge_ocr(base, result)
                if not text.strip():
                    raise IncompleteDocumentError('No readable text recognized on nonblank page')
                info.update(method='native+ocr' if base else 'ocr',
                            ocr_engine=result['engine'], ocr_lines=lines)
                info['warnings'] = ['ocr_may_contain_errors', 'reading_order_is_heuristic']
                info['low_confidence_lines'] = sum(
                    line.get('confidence', 0) < .8 for line in lines)
            else:
                text = native
            texts.append(text)
        except Exception as exc:
            issue = {'page': index+1, 'error': f'{type(exc).__name__}: {exc}'}
            if not options.allow_partial:
                detail = str(exc)
                if hasattr(exc, 'stderr') and exc.stderr:
                    detail += ' ' + exc.stderr.decode('utf-8', errors='replace')[-500:]
                raise IncompleteDocumentError(
                    f'PDF page {index+1} could not be fully parsed: {detail}. '
                    'No version published; configure local OCR or explicitly allow partial parsing.'
                ) from exc
            warnings.append(issue)
            # Never drop the page position, or imply all pages were processed.
            texts.append('')
            info.update(method='failed', error=issue['error'])
        records.append(info)
    offsets, position = [], 0
    for text, record in zip(texts, records, strict=True):
        offsets.append(position)
        record.update(start=position, end=position+len(text))
        for line in record['ocr_lines']:
            line['start'] += position
            line['end'] += position
        position += len(text)+2
    return '\n\n'.join(texts), {
        'parser': 'page-aware-pdf-v1', 'page_starts': offsets, 'pages': records,
        'parse_complete': not warnings, 'parse_warnings': warnings,
        'bbox_coordinates': 'normalized top-left; OCR page/image render',
    }


def parse_image(data, suffix, *, options=None, ocr=None):
    options = options or ParseOptions()
    ocr = ocr or LocalOCR(options.ocr_backend, options.ocr_languages)
    result = ocr.recognize(data, suffix=suffix,
                           timeout=min(options.ocr_timeout_seconds, options.parse_timeout_seconds))
    text, lines = _merge_ocr('', result)
    if not text.strip():
        raise IncompleteDocumentError('Image OCR produced no readable text; visual understanding is not enabled')
    return text, {'parser': 'image-ocr-v1', 'ocr_engine': result['engine'],
                  'parse_complete': True, 'page_starts': [0],
                  'pages': [{'page': 1, 'method': 'ocr', 'start': 0, 'end': len(text), 'ocr_lines': lines}],
                  'bbox_coordinates': 'normalized top-left; OCR page/image render',
                  'parse_warnings': ['ocr_may_contain_errors', 'visual_content_is_not_fully_understood']}
