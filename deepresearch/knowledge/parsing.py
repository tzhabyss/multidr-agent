"""Optional local MinerU 4 CLI adapter. No hosted document upload is performed."""
import shutil
import subprocess
import tempfile
from pathlib import Path


def mineru_parse(data):
    # mineru-kit is stateless; unlike the library CLI it parses the full input.
    executable = shutil.which('mineru-kit')
    if not executable:
        raise RuntimeError('parser=mineru requires a separately installed MinerU 4 mineru-kit and local model weights')
    with tempfile.TemporaryDirectory(prefix='rag-parse-') as directory:
        source, target = Path(directory)/'source.pdf', Path(directory)/'source.md'
        source.write_bytes(data)
        subprocess.run([executable, 'parse', str(source), '-o', str(target), '--tier', 'standard'],
                       check=True, timeout=300, capture_output=True)
        content = target.read_text(encoding='utf-8')
        if not content.strip():
            raise ValueError('MinerU returned empty Markdown')
        return content, {'parser': 'mineru-kit-4', 'page_locations': 'not_available_in_markdown'}
