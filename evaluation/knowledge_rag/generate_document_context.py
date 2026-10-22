"""Generate and freeze corpus-only descriptions BEFORE retrieval evaluation."""
import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

from deepresearch.knowledge.document_context import CONTEXT_PROMPT, PROMPT_SHA256, CONCISE_PROMPT, CONCISE_PROMPT_SHA256, describe_chunk, sha
from deepresearch.knowledge.dual_store import structured_chunks
from deepresearch.llm import OpenAIResearchLLM

ROOT = Path(__file__).parent


async def generate(args):
    # Deliberately do not import load_dataset or open cases.jsonl / evidence labels.
    raw = args.corpus.read_bytes()
    docs = json.loads(raw)
    llm = OpenAIResearchLLM(temperature=0, request_timeout_seconds=args.timeout+2, max_retries=0)
    protocol = {'corpus_sha256': hashlib.sha256(raw).hexdigest(), 'model': llm.model_id,
                'temperature': 0, 'max_output_tokens': 256, 'timeout_seconds': args.timeout,
                'concurrency': args.concurrency, 'prompt': CONTEXT_PROMPT, 'prompt_sha256': PROMPT_SHA256,
                'input_fields': ['document', 'title', 'heading', 'chunk'], 'uses_questions': False,
                'uses_gold': False, 'chunker': 'structured-v1', 'max_context_chars': 180,
                'repair_prompt': CONCISE_PROMPT, 'repair_prompt_sha256': CONCISE_PROMPT_SHA256}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'contexts.json'
    saved = json.loads(path.read_text()) if path.exists() else {'protocol': protocol, 'chunks': {}}
    if saved['protocol'] != protocol:
        raise ValueError('context protocol differs; choose another output directory')
    chunks = [(doc, p) for doc in docs for p in structured_chunks(doc['content'])]
    semaphore = asyncio.Semaphore(args.concurrency)
    async def one(doc, part):
        key = sha(json.dumps([sha(doc['content']), part['start'], part['end']]))
        previous = saved['chunks'].get(key)
        if previous and (previous['status'] == 'generated' or not args.retry_invalid):
            return
        async with semaphore:
            result = await describe_chunk(llm, doc['content'], part, title=doc['title'], timeout=args.timeout,
                                          concise=bool(previous))
            if previous:
                result['previous_attempts'] = previous.get('previous_attempts', []) + [
                    {k: v for k, v in previous.items() if k != 'previous_attempts'}]
            saved['chunks'][key] = {**result, 'document': doc['id'], 'heading': part['path']}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2)+'\n')
            count = len(saved['chunks'])
            if count % 12 == 0 or count == len(chunks):
                print(f'document contexts {count}/{len(chunks)}', flush=True)
    try:
        await asyncio.gather(*(one(d, p) for d, p in chunks))
    finally:
        await llm.client.close()
    attempts = [a for entry in saved['chunks'].values() for a in [*entry.get('previous_attempts', []), entry]]
    freeze = {'contexts_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'protocol': protocol,
              'count': len(saved['chunks']), 'statuses': dict(Counter(e['status'] for e in saved['chunks'].values())),
              'attempt_count': len(attempts),
              'input_tokens': sum(e['input_tokens'] for e in attempts),
              'output_tokens': sum(e['output_tokens'] for e in attempts)}
    (args.output/'context-freeze.json').write_text(json.dumps(freeze, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in freeze.items() if k != 'protocol'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, default=ROOT/'v3/corpus.json')
    parser.add_argument('--output', type=Path, default=ROOT/'document_context')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--retry-invalid', action='store_true')
    asyncio.run(generate(parser.parse_args()))
