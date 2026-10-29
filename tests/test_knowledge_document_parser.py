"""Page routing, provenance and fail-closed publication inputs; OCR quality tested separately."""
import io
import sys
from pathlib import Path

import pytest
from pypdf import PdfWriter

from deepresearch.knowledge import document_parser as parser
from deepresearch.knowledge.document_parser import ParseOptions, IncompleteDocumentError
from deepresearch.knowledge.ocr import LocalOCR, _tesseract_result

FIXTURES = Path(__file__).parent / 'fixtures' / 'knowledge'


class OCR:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def recognize(self, data, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError('OCR backend unavailable')
        return {'engine': 'test', 'lines': [
            {'text': '报销期限十五天。', 'confidence': .95, 'bbox': [.1, .2, .4, .1]},
            {'text': 'Budget 2024: 12345 CNY', 'confidence': .7, 'bbox': [.1, .4, .5, .1]},
        ]}


def blank_pdf(pages=1, password=None):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=600, height=800)
    if password:
        writer.encrypt(password)
    output = io.BytesIO(); writer.write(output)
    return output.getvalue()


def test_mixed_pdf_routes_per_page_and_preserves_exact_spans():
    ocr = OCR()
    text, meta = parser.parse_pdf((FIXTURES/'mixed.pdf').read_bytes(), ocr=ocr)
    assert [c['page'] for c in ocr.calls] == [2]
    assert [p['method'] for p in meta['pages']] == ['native', 'ocr']
    assert meta['parse_complete']
    assert '67890' in text and '12345' in text
    for page in meta['pages']:
        assert meta['page_starts'][page['page']-1] == page['start']
        for line in page['ocr_lines']:
            assert text[line['start']:line['end']] == line['text']
    assert meta['pages'][1]['low_confidence_lines'] == 1


def test_missing_ocr_fails_with_page_number():
    with pytest.raises(IncompleteDocumentError, match='page 2.*OCR backend unavailable'):
        parser.parse_pdf((FIXTURES/'mixed.pdf').read_bytes(), ocr=OCR(fail=True))


def test_partial_requires_opt_in_and_preserves_failed_page():
    text, meta = parser.parse_pdf((FIXTURES/'mixed.pdf').read_bytes(),
        options=ParseOptions(allow_partial=True), ocr=OCR(fail=True))
    assert not meta['parse_complete'] and meta['parse_warnings'][0]['page'] == 2
    assert meta['pages'][1]['method'] == 'failed'
    assert text[meta['page_starts'][1]:] == ''


def test_native_only_cannot_silently_publish_scanned_pdf():
    with pytest.raises(IncompleteDocumentError, match='parser=native'):
        parser.parse_pdf((FIXTURES/'mixed.pdf').read_bytes(), native_only=True)


def test_blank_pages_do_not_call_ocr():
    ocr = OCR(fail=True)
    text, meta = parser.parse_pdf(blank_pdf(2), ocr=ocr)
    assert text == '\n\n' and not ocr.calls
    assert [p['method'] for p in meta['pages']] == ['blank', 'blank']


def test_page_limit_and_encryption_rejected():
    with pytest.raises(ValueError, match='page parsing limit'):
        parser.parse_pdf(blank_pdf(2), options=ParseOptions(max_pages=1))
    with pytest.raises(ValueError, match='password'):
        parser.parse_pdf(blank_pdf(password='secret'))


@pytest.mark.parametrize('native', ['短的原文123。', 'Enough native text. ' * 6])
def test_mixed_page_keeps_exact_native_and_merges_ocr(monkeypatch, native):
    monkeypatch.setattr(parser, '_native_page', lambda p: (native, .8, True, True))
    ocr = OCR()
    text, meta = parser.parse_pdf(blank_pdf(), ocr=ocr)
    assert text.startswith(native.rstrip()) and '十五天' in text
    assert meta['pages'][0]['method'] == 'native+ocr'


def test_corrupt_layer_replaced_and_extraction_errors_fall_back(monkeypatch):
    monkeypatch.setattr(parser, '_native_page', lambda p: ('\ue001'*60, 0, False, True))
    text, meta = parser.parse_pdf(blank_pdf(), ocr=OCR())
    assert '\ue001' not in text and meta['pages'][0]['method'] == 'ocr'
    def broken(p):
        raise ValueError('bad font map')
    monkeypatch.setattr(parser, '_native_page', broken)
    text, meta = parser.parse_pdf(blank_pdf(), ocr=OCR())
    assert '十五天' in text and 'bad font map' in meta['pages'][0]['native_error']


def test_clean_short_native_without_images_does_not_need_ocr(monkeypatch):
    monkeypatch.setattr(parser, '_native_page', lambda p: ('标题123', 0, False, True))
    ocr = OCR(fail=True)
    text, meta = parser.parse_pdf(blank_pdf(), ocr=ocr)
    assert text == '标题123' and not ocr.calls


def test_merge_deduplicates_and_offsets_are_exact():
    text, lines = parser._merge_ocr('Budget 2024: 12345 CNY', OCR().recognize(b''))
    assert text.count('12345') == 1 and len(lines) == 1
    assert text[lines[0]['start']:lines[0]['end']] == '报销期限十五天。'


def test_empty_ocr_nonblank_page_is_rejected(monkeypatch):
    monkeypatch.setattr(parser, '_native_page', lambda p: ('', 1, True, True))
    ocr = OCR()
    ocr.recognize = lambda *a, **kw: {'engine': 'empty', 'lines': []}
    with pytest.raises(IncompleteDocumentError, match='No readable text'):
        parser.parse_pdf(blank_pdf(), ocr=ocr)
    with pytest.raises(IncompleteDocumentError, match='no readable text'):
        parser.parse_image(b'image', '.png', ocr=ocr)


def test_document_deadline_applies_before_each_page():
    with pytest.raises(IncompleteDocumentError, match='deadline'):
        parser.parse_pdf(blank_pdf(), options=ParseOptions(parse_timeout_seconds=-1))


def test_image_metadata_and_disabled_backend():
    text, meta = parser.parse_image(b'image', '.png', ocr=OCR())
    assert meta['parse_complete'] and meta['pages'][0]['ocr_lines'][0]['start'] == 0
    assert text.startswith('报销')
    with pytest.raises(RuntimeError, match='disabled'):
        LocalOCR('disabled').recognize(b'image', suffix='.png')


def test_tesseract_tsv_lines_and_coordinates():
    tsv = ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
           '1\t1\t0\t0\t0\t0\t0\t0\t100\t200\t-1\t\n'
           '5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t90\tBudget\n'
           '5\t1\t1\t1\t1\t2\t45\t20\t20\t10\t80\t12345\n')
    result = _tesseract_result(tsv)
    assert result['lines'][0]['text'] == 'Budget 12345'
    assert result['lines'][0]['bbox'] == [.1, .1, .55, .05]
    assert result['lines'][0]['confidence'] == .85
    with pytest.raises(ValueError, match='Multi-frame'):
        _tesseract_result(tsv+'1\t2\t0\t0\t0\t0\t0\t0\t100\t200\t-1\t\n')


@pytest.mark.skipif(sys.platform != 'darwin', reason='requires Apple Vision and Swift tools')
def test_real_local_vision_image_and_mixed_pdf():
    # A real generated Chinese/English scan, not a mocked recognition response.
    text, meta = parser.parse_image((FIXTURES/'scan.png').read_bytes(), '.png')
    assert '研究资料扫描件测试' in text and '12345' in text and 'deepresearch' in text
    assert meta['ocr_engine'] == 'apple-vision'
    text, meta = parser.parse_pdf((FIXTURES/'mixed.pdf').read_bytes())
    assert [p['method'] for p in meta['pages']] == ['native', 'ocr']
    assert '67890' in text and '12345' in text and '研究资料扫描件测试' in text
