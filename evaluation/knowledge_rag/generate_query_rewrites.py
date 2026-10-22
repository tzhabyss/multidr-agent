"""Freeze query-only LLM rewrites before running any retrieval/label scoring."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from deepresearch.knowledge.query_rewrite import PROMPT_SHA256, REWRITE_PROMPT, rewrite_query
from deepresearch.llm import OpenAIResearchLLM

ROOT = Path(__file__).parent
OUT = ROOT/'query_rewrite'


async def generate(args):
    manifest = json.loads((ROOT/'v3/manifest.json').read_text())
    data = (ROOT/'v3/cases.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest['cases_sha256']
    # No corpus file is opened. Only these question strings are sent to the LLM.
    cases = [json.loads(line) for line in data.decode().splitlines() if line]
    queries = list(dict.fromkeys(q for case in cases for q in (
        case['subqueries'] if case['group']=='multi_source' else [case['query']])))
    llm = OpenAIResearchLLM(temperature=0, request_timeout_seconds=args.timeout+2, max_retries=0)
    protocol = {'model': llm.model_id, 'temperature': 0, 'max_output_tokens': 256,
                'timeout_seconds': args.timeout, 'concurrency': args.concurrency,
                'prompt_sha256': PROMPT_SHA256, 'prompt': REWRITE_PROMPT,
                'cases_sha256': manifest['cases_sha256'], 'question_count': len(queries),
                'input_fields': ['question'], 'uses_corpus': False, 'uses_gold': False}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'rewrites.json'
    saved = json.loads(path.read_text()) if path.exists() else {'protocol': protocol, 'queries': {}}
    if saved['protocol'] != protocol:
        raise ValueError('cached rewrite protocol differs; use another output directory')
    # Keep cache insertion order: merely resuming a completed run must not
    # change the frozen byte hash or invalidate the downstream paired results.
    if not set(saved['queries']).issubset(set(queries)):
        raise ValueError('cached queries do not match the frozen question set')
    semaphore = asyncio.Semaphore(args.concurrency)
    async def one(query):
        if query in saved['queries']: return
        async with semaphore:
            result = await rewrite_query(llm, query, timeout=args.timeout)
            saved['queries'][query] = result
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2)+'\n')
            count = len(saved['queries'])
            if count%12==0 or count==len(queries):
                print(f'rewrite {count}/{len(queries)}', flush=True)
    try:
        await asyncio.gather(*(one(q) for q in queries))
    finally:
        await llm.client.close()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output/'rewrite-freeze.json').write_text(json.dumps({'rewrites_sha256': sha,
        'protocol': protocol, 'statuses': {s: sum(r['status']==s for r in saved['queries'].values())
            for s in sorted({r['status'] for r in saved['queries'].values()})}}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'frozen_sha256': sha, 'count': len(saved['queries']),
                     'statuses': json.loads((args.output/'rewrite-freeze.json').read_text())['statuses']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--concurrency', type=int, default=4)
    asyncio.run(generate(parser.parse_args()))
