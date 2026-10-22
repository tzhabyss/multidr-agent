"""Fault-injection audit of the actual local RAG ingestion implementation.

SQLite transactions are tested directly. This does not simulate Mongo/Milvus
transactions or claim those services are installed. Known requirement gaps are
reported as unmet instead of being hidden behind xfail or a retrieval score.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from deepresearch.knowledge import KnowledgeStore
from deepresearch.knowledge.store import chunks, digest
from deepresearch.memory_system.embeddings import HashEmbeddingProvider, ResilientEmbeddingProvider

OUT=Path(__file__).parent/'ingestion_audit'


class CountingProvider(HashEmbeddingProvider):
    def __init__(self, dimensions=128):
        super().__init__(dimensions)
        self.calls=0;self.items=0
    def encode_documents(self,values):
        self.calls+=1;self.items+=len(values)
        return super().encode_documents(values)


class AuditFailure(AssertionError):
    def __init__(self, message, observed=None):
        super().__init__(message);self.observed=observed or {}


def require(condition,message,observed=None):
    if not condition:raise AuditFailure(message,observed)


def health(store):
    with store.connect() as db:
        return {'snapshots':db.execute('SELECT count(*) FROM snapshots').fetchone()[0],
                'active':db.execute("SELECT count(*) FROM snapshots WHERE status='active'").fetchone()[0],
                'chunks':db.execute('SELECT count(*) FROM chunks').fetchone()[0],
                'orphan_chunks':db.execute('SELECT count(*) FROM chunks c LEFT JOIN snapshots s ON s.id=c.snapshot WHERE s.id IS NULL').fetchone()[0],
                'integrity':db.execute('PRAGMA integrity_check').fetchone()[0]}


def make_pdf(path,pages):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
    writer=PdfWriter()
    for text in pages:
        page=writer.add_blank_page(400,400)
        font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream=DecodedStreamObject()
        escaped=text.replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
        stream.set_data(f'BT /F1 12 Tf 30 360 Td ({escaped}) Tj ET'.encode('ascii'))
        page[NameObject('/Contents')]=writer._add_object(stream)
    with path.open('wb') as output:writer.write(output)


def run(output=OUT/'results.json',real_model=True):
    results=[]
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        def execute(name,category,fn,severity='medium',requirement=''):
            start=time.perf_counter()
            workspace=root/f'{len(results):02}'
            workspace.mkdir()
            try:
                observed=fn(workspace) or {}
                row={'name':name,'category':category,'requirement':requirement,'status':'pass','observed':observed}
            except AuditFailure as exc:
                row={'name':name,'category':category,'requirement':requirement,'status':'unmet','severity':severity,
                     'reason':str(exc),'observed':exc.observed}
            except Exception as exc:
                row={'name':name,'category':category,'requirement':requirement,'status':'error','severity':severity,
                     'reason':f'{type(exc).__name__}: {exc}'}
            row['seconds']=time.perf_counter()-start
            results.append(row)
            print(f"{len(results):02} {row['status']}: {name}",flush=True)
        def store(p,provider=None,contextual=False):
            return KnowledgeStore(p/'db.sqlite',provider=provider or CountingProvider(),contextual=contextual)
        def seed(p,provider=None,contextual=False):
            db=store(p,provider,contextual)
            item=db.ingest('旧版费用为100元。',uri='file:///report.md',title='报告')
            return db,item['snapshot_id']
        def unchanged(db,sid,before):
            after=health(db)
            require(after==before,'failure left partial records',{'before':before,'after':after})
            require(db.snapshot(sid,collection='library')['status']=='active','old version archived on failure')
            return after

        def duplicate(p,contextual=False):
            provider=CountingProvider();db=store(p,provider,contextual)
            first=db.ingest('费用100元。',uri='file:///a.md',title='A');calls=provider.calls
            second=db.ingest('费用100元。',uri='file:///a.md',title='A')
            require(first['snapshot_id']==second['snapshot_id'] and second['duplicate'],'duplicate snapshot was inserted')
            require(provider.calls==calls,'duplicate re-embedded',{'embedding_calls_before':calls,'after':provider.calls})
            return {'initial_embedding_batches':calls,'duplicate_extra_batches':provider.calls-calls,**health(db)}
        execute('same_uri_same_hash_skips_embedding','dedup',duplicate,requirement='同来源相同内容不重复入库／编码')
        execute('contextual_duplicate_skips_both_views','dedup',lambda p:duplicate(p,True))
        def cross_uri(p):
            provider=CountingProvider();db=store(p,provider)
            db.ingest('费用100元。',uri='file:///a.md',title='A');before=provider.calls
            db.ingest('费用100元。',uri='file:///copy.md',title='Copy')
            observed={'duplicate_extra_embedding_batches':provider.calls-before,**health(db)}
            require(provider.calls==before,'same content at another URI is re-embedded; no collection-wide content reuse',observed)
            return observed
        execute('same_hash_different_uri_reuses_embedding','dedup',cross_uri,requirement='同哈希副本复用向量，保留不同来源身份')
        def scoped(p):
            db=store(p)
            a=db.ingest('费用100元。',uri='file:///a.md',title='A',collection='a')
            b=db.ingest('费用100元。',uri='file:///a.md',title='A',collection='b')
            require(a['snapshot_id']!=b['snapshot_id'],'collections shared identity')
            require(not db.search('费用',collection='c',mode='bm25'),'cross-collection leak')
            return health(db)
        execute('collection_isolation','dedup',scoped,'high')
        def whitespace(p):
            db=store(p)
            a=db.ingest('费用100元。',uri='file:///a.md',title='A')
            b=db.ingest('费用 100元。',uri='file:///a.md',title='A')
            require(a['snapshot_id']!=b['snapshot_id'],'raw parsed text was normalized before SHA')
            return health(db)
        execute('raw_parsed_text_sha_distinguishes_changes','dedup',whitespace)
        def metadata(p):
            db=store(p)
            db.ingest('费用100元。',uri='file:///a.md',title='Old',metadata={'published_at':'2024-01-01'})
            item=db.ingest('费用100元。',uri='file:///a.md',title='Corrected',metadata={'published_at':'2024-02-01'})
            value=db.snapshot(item['snapshot_id'],collection='library')
            observed={'duplicate':item['duplicate'],'stored_title':value['title'],'stored_metadata':json.loads(value['metadata'])}
            require(value['title']=='Corrected','duplicate path discards title/metadata corrections',observed)
            return observed
        execute('duplicate_metadata_correction','dedup',metadata,requirement='正文不变时允许修正元数据而无需重编码')

        def text_file(p):
            path=p/'含 空格.md';text='# 标题\n\n费用为128元。\n\nUnicode Ω emoji 🌲'
            path.write_text(text);db=store(p);item=db.ingest_file(path)
            snap=db.snapshot(item['snapshot_id'],collection='library')
            require(snap['content']==text and snap['uri']==path.resolve().as_uri(),'text/URI altered')
            require(snap['hash']==digest(text),'SHA mismatch')
            return {'chars':len(text),'hash_verified':True}
        execute('utf8_markdown_and_file_uri','parsing',text_file)
        def pdf(p):
            path=p/'pages.pdf';make_pdf(path,['Alpha fee 123 USD.','Beta duration 47 days.'])
            db=store(p);item=db.ingest_file(path)
            hits=db.search('Beta duration',mode='bm25')
            hit=next(h for h in hits if 'Beta' in h['content'])
            require(hit['page']==2,'PDF page locator wrong',{'page':hit['page']})
            require(len(json.loads(db.snapshot(item['snapshot_id'],collection='library')['metadata'])['page_starts'])==2,'PDF pages missing')
            return {'page_count':2,'beta_page':hit['page'],**health(db)}
        execute('two_page_pdf_text_and_page_locations','parsing',pdf,'high')
        def pdf_dedup(p):
            import pypdf
            path=p/'same.pdf';make_pdf(path,['Fee 123 USD.'])
            provider=CountingProvider();db=store(p,provider)
            actual=pypdf.PdfReader;calls=[]
            def counted(*args,**kwargs):calls.append(1);return actual(*args,**kwargs)
            pypdf.PdfReader=counted
            try:
                db.ingest_file(path);before=provider.calls;db.ingest_file(path)
            finally:pypdf.PdfReader=actual
            observed={'parser_calls':len(calls),'duplicate_embedding_calls':provider.calls-before}
            require(len(calls)==1,'duplicate file is parsed again; SHA dedup happens after parsing',observed)
            return observed
        execute('file_sha_before_reparsing','parsing',pdf_dedup,requirement='上传字节哈希命中时跳过解析与编码')
        def bad_file(p,kind):
            db,sid=seed(p);before=health(db)
            path=p/f'bad.{kind}'
            if kind=='pdf':path.write_bytes(b'not a PDF')
            elif kind=='blankpdf':
                path=p/'blank.pdf';make_pdf(path,[''])
            elif kind=='docx':path.write_bytes(b'not a docx')
            else:path.write_text(' \n\t')
            try:db.ingest_file(path)
            except Exception as exc:
                return {'exception':type(exc).__name__,**unchanged(db,sid,before)}
            raise AuditFailure('invalid input was published')
        for kind in ['pdf','blankpdf','txt','docx']:
            execute(f'reject_invalid_or_unsupported_{kind}','parsing',lambda p,k=kind:bad_file(p,k))
        def source_persistence(p):
            path=p/'source.txt';path.write_text('费用128元。');db=store(p);item=db.ingest_file(path);path.unlink()
            require(db.snapshot(item['snapshot_id'],collection='library')['content']=='费用128元。','source deletion lost canonical text')
            return {'text_persists_after_file_removed':True}
        execute('parsed_snapshot_survives_source_deletion','parsing',source_persistence)

        def offsets(p):
            text='# Root\n\n## Section\n\n'+('中文费用128元。English line.\n'*200)
            parts=chunks(text);covered=set()
            for part in parts:
                require(0<=part['start']<part['end']<=len(text),'invalid span')
                require(part['end']-part['start']<=600,'chunk exceeds 600 chars')
                covered.update(range(part['start'],part['end']))
            missing=[i for i,ch in enumerate(text) if not ch.isspace() and i not in covered]
            require(not missing,'nonwhitespace source characters lost',{'missing':len(missing)})
            require(parts[0]['path']==['Root','Section'],'heading path lost')
            return {'chunks':len(parts),'nonwhitespace_missing':0,'max_chars':max(x['end']-x['start'] for x in parts)}
        execute('chunk_offsets_hierarchy_and_content_coverage','chunking',offsets,'high')
        def long_heading(p):
            text='# '+('标题'*500)
            parts=chunks(text)
            observed={'chunks':len(parts),'max_chars':max(x['end']-x['start'] for x in parts)}
            require(all(x['end']-x['start']<=600 for x in parts),'heading-only input bypasses chunk size bound',observed)
            return observed
        execute('heading_only_length_bound','chunking',long_heading)
        def table_headers(p):
            header='| 项目 | 数值 |'
            text='# 成本表\n\n'+header+'\n|---|---|\n'+'\n'.join(f'| 成本项目{i:03} | {i+100}万元 |' for i in range(100))
            parts=chunks(text);table_parts=[x for x in parts if '成本项目' in text[x['start']:x['end']]]
            missing=sum(header not in text[x['start']:x['end']] for x in table_parts)
            observed={'table_chunks':len(table_parts),'chunks_without_header':missing}
            require(missing==0,'long table is split without carrying its header to later chunks',observed)
            return observed
        execute('long_table_header_preservation','chunking',table_headers,requirement='跨片表格保留表头和单位')

        def batching(p):
            provider=CountingProvider();db=store(p,provider)
            text=('独立段落费用128元。\n\n'*120)
            item=db.ingest(text,uri='file:///batch.md',title='Batch')
            require(provider.calls==1 and provider.items==item['chunks'],'encoding not batched')
            return {'chunks':item['chunks'],'embedding_batches':provider.calls,'items':provider.items}
        execute('document_chunks_encoded_as_batch','embedding',batching)
        def invalid_vectors(p,kind):
            db,sid=seed(p);before=health(db)
            class Invalid(CountingProvider):
                # Simulate a provider claiming healthy semantic output, not an
                # explicitly lexical-only degraded index.
                @property
                def degraded(self):return False
                @property
                def model_id(self):return 'fault-injection-semantic-128'
                def encode_documents(self,values):
                    if kind=='short_batch':return super().encode_documents(values)[:-1]
                    if kind=='empty':return [[] for _ in values]
                    if kind=='nan':return [[float('nan')]*self.dimensions for _ in values]
                    if kind=='dimension':return [[1.,0.,0.] for _ in values]
                    if kind=='infinity':return [[float('inf')]*self.dimensions for _ in values]
                    if kind=='zero':return [[0.]*self.dimensions for _ in values]
                    if kind=='nonnumeric':return [['oops']*self.dimensions for _ in values]
            db.provider=Invalid()
            try:db.ingest('新版费用200元。',uri='file:///report.md',title='报告')
            except Exception as exc:return {'exception':type(exc).__name__,**unchanged(db,sid,before)}
            with db.connect() as conn:
                published=conn.execute("SELECT degraded,model FROM snapshots WHERE status='active'").fetchone()
            raise AuditFailure(f'{kind} vectors were published active',
                {**health(db),'published_degraded':bool(published['degraded']),'model':published['model']})
        for kind in ['short_batch','empty','nan','dimension','infinity','zero','nonnumeric']:
            execute(f'reject_{kind}_embeddings','embedding',lambda p,k=kind:invalid_vectors(p,k),'high')
        def raw_failure(p):
            db,sid=seed(p,contextual=True);before=health(db)
            class FailSecond(CountingProvider):
                def encode_documents(self,values):
                    if self.calls==1:raise RuntimeError('second view failure')
                    return super().encode_documents(values)
            db.provider=FailSecond()
            try:db.ingest('新版费用200元。',uri='file:///report.md',title='报告')
            except RuntimeError:return unchanged(db,sid,before)
            raise AuditFailure('expected raw-view failure missing')
        execute('second_embedding_view_failure_keeps_old','embedding',raw_failure,'high')

        def publish(p):
            db,old=seed(p)
            new=db.ingest('新版费用200元。',uri='file:///report.md',title='报告')['snapshot_id']
            oldrow=db.snapshot(old,collection='library');newrow=db.snapshot(new,collection='library')
            require(oldrow['status']=='archived' and newrow['status']=='active' and newrow['supersedes']==old,'version chain wrong')
            require(health(db)['active']==1,'multiple active versions')
            require('200' in db.search('费用',mode='bm25')[0]['content'],'default retrieval returned old version')
            return {'supersedes_correct':True,**health(db)}
        execute('publish_complete_new_version_then_archive_old','version',publish,'high')
        def sql_failure(p,stage):
            db,sid=seed(p);before=health(db)
            with db.connect() as conn:
                if stage=='snapshot':
                    conn.execute("CREATE TRIGGER fail BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT,'chunk write failed'); END")
                elif stage=='partial_chunks':
                    conn.execute("CREATE TRIGGER fail BEFORE INSERT ON chunks WHEN NEW.start>0 BEGIN SELECT RAISE(ABORT,'partial write failed'); END")
                else:
                    conn.execute("CREATE TRIGGER fail BEFORE UPDATE OF status ON snapshots WHEN NEW.status='archived' BEGIN SELECT RAISE(ABORT,'archive failed'); END")
            try:db.ingest(('新版费用200元。\n\n'*120),uri='file:///report.md',title='报告')
            except sqlite3.DatabaseError:return unchanged(db,sid,before)
            raise AuditFailure('injected storage failure did not occur')
        for stage in ['snapshot','partial_chunks','archive']:
            execute(f'rollback_failure_at_{stage}','rollback',lambda p,s=stage:sql_failure(p,s),'critical')
        def historical(p):
            db,sid=seed(p);cutoff=db.snapshot(sid,collection='library')['created_at']
            db.ingest('新版费用200元。',uri='file:///report.md',title='报告')
            hits=db.search('费用',as_of=cutoff,mode='bm25')
            require(hits and all(h['snapshot_id']==sid for h in hits),'historical capture cutoff wrong')
            item=db.ingest('旧版费用为100元。',uri='file:///report.md',title='报告')
            require(item['duplicate'] and item['status']=='archived','old reimport reactivated version')
            require(health(db)['active']==1,'version count invalid')
            return {'historical_hits':len(hits),'old_reimport_status':item['status']}
        execute('capture_time_history_and_archived_reimport','version',historical)
        def renamed(p):
            db,sid=seed(p)
            db.ingest('新版费用200元。',uri='file:///renamed-report.md',title='报告')
            observed=health(db)
            require(observed['active']==1,'version identity is URI-based; renamed document becomes another active document',observed)
            return observed
        execute('renamed_logical_document_version_detection','version',renamed,requirement='跨文件名识别同一份制度的新版本')
        def job_record(p):
            db,sid=seed(p)
            with db.connect() as conn:
                names=[row['name'] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            observed={'tables':names,'visible_statuses':['active','archived']}
            require(any('job' in n or 'outbox' in n for n in names),'no durable ingestion job/pending/failed audit or retry queue',observed)
            return observed
        execute('durable_ingestion_job_and_retry_record','version',job_record,requirement='失败作业可追踪与重试；不等于需要分布式事务')
        def visibility(p):
            db,sid=seed(p);entered=threading.Event();release=threading.Event()
            class Paused(KnowledgeStore):
                @contextmanager
                def connect(self):
                    with super().connect() as conn:
                        class Proxy:
                            def __getattr__(self,n):return getattr(conn,n)
                            def execute(self,sql,*args):
                                out=conn.execute(sql,*args)
                                if sql.startswith('INSERT INTO chunks'):
                                    entered.set()
                                    if not release.wait(5):raise TimeoutError('reader check did not release writer')
                                return out
                        yield Proxy()
            writer=Paused(db.path,provider=CountingProvider())
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(writer.ingest,'新版费用200元。',uri='file:///report.md',title='报告')
                try:
                    require(entered.wait(5),'write pause not reached')
                    checks=[]
                    for _ in range(20):
                        with db.connect() as conn:
                            active=conn.execute("SELECT id FROM snapshots WHERE status='active'").fetchall()
                        checks.append([r['id'] for r in active]==[sid])
                    require(all(checks),'reader saw empty or partially published version')
                finally:release.set()
                result=future.result(timeout=5)
            require(db.snapshot(sid,collection='library')['status']=='archived','old version not archived after commit')
            return {'precommit_reads':len(checks),'old_visible_during_all_reads':True,'new_version_ready':bool(result['snapshot_id'])}
        execute('readers_see_old_version_until_commit','concurrency',visibility,'critical')

        def concurrent(p,separate=False):
            barrier=threading.Barrier(2) if separate else None
            class Racing(CountingProvider):
                def encode_documents(self,values):
                    result=super().encode_documents(values)
                    if barrier:barrier.wait(timeout=5)
                    return result
            db=store(p,CountingProvider())
            if separate:
                providers=[Racing(),Racing()]
                stores=[KnowledgeStore(db.path,provider=provider) for provider in providers]
            else:
                providers=[db.provider];stores=[db,db]
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(lambda s:s.ingest('费用100元。',uri='file:///same.md',title='Same'),stores))
            observed={'insertions':sum(not r['duplicate'] for r in results),
                      'embedding_batches':sum(p.calls for p in providers),**health(db)}
            require(observed['insertions']==1 and observed['snapshots']==1,'concurrent dedup corrupted rows',observed)
            if separate:
                require(observed['embedding_batches']==1,'separate workers both embed before transactional dedup recheck',observed)
            return observed
        execute('same_instance_concurrent_dedup','concurrency',concurrent,'high')
        execute('separate_instances_avoid_duplicate_embedding','concurrency',lambda p:concurrent(p,True))
        def concurrent_versions(p):
            db,sid=seed(p)
            writers=[KnowledgeStore(db.path,provider=CountingProvider()) for _ in range(2)]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(w.ingest,f'并发版本费用{n}元。',uri='file:///report.md',title='报告') for w,n in zip(writers,[200,300])]
                for f in futures:f.result(timeout=5)
            with db.connect() as conn:
                rows=[dict(row) for row in conn.execute('SELECT id,supersedes,status FROM snapshots')]
            require(sum(r['status']=='active' for r in rows)==1,'multiple active versions')
            active=next(r for r in rows if r['status']=='active');byid={r['id']:r for r in rows};chain=[]
            while active:
                chain.append(active['id']);active=byid.get(active['supersedes'])
            require(len(chain)==3 and len(set(chain))==3,'broken supersedes chain')
            return {'chain_length':len(chain),**health(db)}
        execute('concurrent_different_versions_form_one_chain','concurrency',concurrent_versions,'high')
        def hard_crash(p):
            db,sid=seed(p);before=health(db)
            script=r'''import os,sys
from contextlib import contextmanager
from deepresearch.knowledge import KnowledgeStore
from deepresearch.memory_system.embeddings import HashEmbeddingProvider
class Crashing(KnowledgeStore):
 @contextmanager
 def connect(self):
  with super().connect() as conn:
   conn.create_function('hard_exit',0,lambda:os._exit(23))
   conn.execute('CREATE TEMP TRIGGER die BEFORE INSERT ON chunks BEGIN SELECT hard_exit(); END')
   yield conn
store=Crashing(sys.argv[1],provider=HashEmbeddingProvider())
store.ingest('新版费用200元。',uri='file:///report.md',title='报告')
'''
            child=subprocess.run([sys.executable,'-c',script,str(db.path)],capture_output=True,timeout=15)
            require(child.returncode==23,'process did not crash at write boundary',{'exit_code':child.returncode})
            return {'child_exit':child.returncode,**unchanged(db,sid,before)}
        execute('process_exit_mid_write_recovers_without_orphans','recovery',hard_crash,'critical')
        def reindex_rollback(p):
            db,sid=seed(p);db.ingest(('其他资料。\n\n'*100),uri='file:///b.md',title='B')
            with db.connect() as conn:
                before=[tuple(row) for row in conn.execute('SELECT id,vector FROM chunks ORDER BY id')]
                conn.execute("CREATE TRIGGER fail BEFORE INSERT ON chunks WHEN NEW.start>0 BEGIN SELECT RAISE(ABORT,'rebuild failed'); END")
            db.provider=CountingProvider(256)
            try:db.reindex()
            except sqlite3.DatabaseError:
                with db.connect() as conn:after=[tuple(row) for row in conn.execute('SELECT id,vector FROM chunks ORDER BY id')]
                require(after==before,'partial reindex was committed')
                return {'unchanged_chunk_rows':len(after),**health(db)}
            raise AuditFailure('expected rebuild fault missing')
        execute('reindex_failure_rolls_back_entire_projection','recovery',reindex_rollback,'critical')
        def restart(p):
            db,sid=seed(p)
            reopened=KnowledgeStore(db.path,provider=CountingProvider())
            require(reopened.search('费用',mode='bm25')[0]['snapshot_id']==sid,'restart lost index')
            return health(reopened)
        execute('restart_persistence_and_retrieval','recovery',restart,'high')
        def migration_race(p):
            # Existing original schema with no derived-context columns.
            path=p/'legacy.sqlite'
            with sqlite3.connect(path) as conn:
                conn.executescript("""CREATE TABLE snapshots(id TEXT PRIMARY KEY,collection TEXT,uri TEXT,title TEXT,content TEXT,
                    hash TEXT,created_at TEXT,status TEXT,supersedes TEXT,model TEXT,degraded INTEGER,metadata TEXT);
                    CREATE TABLE chunks(id TEXT PRIMARY KEY,snapshot TEXT,start INTEGER,end INTEGER,heading TEXT,vector TEXT);""")
            barrier=threading.Barrier(2)
            class Migrating(KnowledgeStore):
                @contextmanager
                def connect(self):
                    with super().connect() as conn:
                        class Proxy:
                            def __getattr__(self,n):return getattr(conn,n)
                            def execute(self,sql,*args):
                                out=conn.execute(sql,*args)
                                if sql=='PRAGMA table_info(snapshots)':
                                    rows=list(out)
                                    barrier.wait(timeout=5)
                                    return rows
                                return out
                        yield Proxy()
            def initialize(_):
                try:
                    Migrating(path,provider=CountingProvider())
                    return 'ok'
                except Exception as exc:return f'{type(exc).__name__}: {exc}'
            with ThreadPoolExecutor(max_workers=2) as pool:
                observed={'initializations':list(pool.map(initialize,range(2)))}
            require(all(x=='ok' for x in observed['initializations']),'concurrent schema migration is check-then-alter without synchronization',observed)
            return observed
        execute('concurrent_legacy_schema_initialization','concurrency',migration_race,'high')

        def checksum(p):
            db,sid=seed(p)
            with db.connect() as conn:conn.execute("UPDATE snapshots SET content='tampered' WHERE id=?",(sid,))
            try:db.snapshot(sid,collection='library')
            except ValueError:return {'corruption_detected':True}
            raise AuditFailure('snapshot hash corruption not detected')
        execute('immutable_text_checksum_verification','recovery',checksum,'high')
        def rebuild(p):
            db,sid=seed(p);before=db.snapshot(sid,collection='library')
            with db.connect() as conn:conn.execute('DELETE FROM chunks')
            require(not db.search('费用',mode='bm25'),'deleted index still returned candidates')
            result=db.reindex();after=db.snapshot(sid,collection='library')
            require(before['hash']==after['hash'] and before['status']==after['status'],'rebuild altered fact plane')
            require(db.search('费用',mode='bm25'),'rebuild did not restore recall')
            return {**result,**health(db)}
        execute('rebuild_lost_index_from_canonical_text','recovery',rebuild,'high')

        if real_model:
            def real(p):
                provider=ResilientEmbeddingProvider(fallback_enabled=False)
                db=store(p,provider)
                start=time.perf_counter()
                item=db.ingest('# 研发报告\n\n研发投入为128万元。\n\n试验样本量为300例。',uri='file:///real.md',title='真实模型测试')
                first_ms=(time.perf_counter()-start)*1000
                with db.connect() as conn:
                    rows=list(conn.execute('SELECT vector,raw_vector FROM chunks'))
                dimensions=[len(json.loads(r['vector'])) for r in rows]
                require(not provider.degraded and all(n==provider.dimensions for n in dimensions),'real embedding dimensions/degradation invalid')
                start=time.perf_counter();again=db.ingest('# 研发报告\n\n研发投入为128万元。\n\n试验样本量为300例。',uri='file:///real.md',title='真实模型测试')
                duplicate_ms=(time.perf_counter()-start)*1000
                require(again['duplicate'],'real-model dedup failed')
                hits=db.search('研发投入金额',mode='vector')
                require(hits and any('128' in h['content'] for h in hits),'real vector index not queryable')
                return {'model':provider.model_id,'degraded':provider.degraded,'dimensions':provider.dimensions,
                        'chunks':item['chunks'],'initial_ms_including_load':first_ms,'duplicate_ms':duplicate_ms,'vector_search_hits':len(hits)}
            execute('real_e5_ingestion_duplicate_and_vector_query','real_model',real,'high')
            def token_budget(p):
                provider=ResilientEmbeddingProvider(fallback_enabled=False)
                text=('# 研究记录\n\n'+('研发投入统计必须保留公司名称、报告年份、金额单位、统计口径及完整表格脚注。'*80))
                db=store(p,provider)
                item=db.ingest(text,uri='file:///tokens.md',title='编码上限测试')
                model=provider._active._load()
                with db.connect() as conn:
                    parts=[dict(r) for r in conn.execute('SELECT start,end FROM chunks ORDER BY start')]
                lengths=[len(model.tokenizer(text[x['start']:x['end']],truncation=False)['input_ids']) for x in parts]
                limit=model.max_seq_length
                observed={'model':provider.model_id,'model_max_tokens':limit,'chunks':len(parts),
                          'oversized_chunks':sum(n>limit for n in lengths),'max_actual_tokens':max(lengths)}
                require(not observed['oversized_chunks'],'character-based chunks exceed actual embedding token limit and are silently truncated',observed)
                return observed
            execute('real_e5_chunk_token_budget','real_model',token_budget,'high')
            def throughput(p):
                provider=ResilientEmbeddingProvider(fallback_enabled=False)
                db=store(p,provider);measurements=[]
                for size in [1000,10000,50000]:
                    unit='设备维护记录需保留温度单位和完整检查步骤。\n\n'
                    text=(unit*(size//len(unit)+1))[:size]
                    uri=f'file:///size-{size}.md'
                    start=time.perf_counter();item=db.ingest(text,uri=uri,title='吞吐测试');seconds=time.perf_counter()-start
                    start=time.perf_counter();duplicate=db.ingest(text,uri=uri,title='吞吐测试');dup_ms=(time.perf_counter()-start)*1000
                    require(duplicate['duplicate'],'size test duplicate failed')
                    measurements.append({'chars':size,'chunks':item['chunks'],'ingest_seconds':seconds,
                                         'chars_per_second':size/max(seconds,.000001),'duplicate_ms':dup_ms})
                with db.connect() as conn:
                    copies=conn.execute('SELECT count(*) AS total,sum(vector=raw_vector) AS duplicated FROM chunks').fetchone()
                return {'model':provider.model_id,'sizes':measurements,'database_bytes':db.path.stat().st_size,
                        'total_chunk_rows':copies['total'],'identical_vector_and_raw_vector_rows':copies['duplicated'],
                        'note':'Short synthetic repeated paragraphs; not production PDF throughput. First size includes lazy model load.'}
            execute('real_e5_ingestion_size_and_duplicate_latency','performance',throughput)

        report={'scope':'RAG ingestion engineering, local SQLite implementation',
                'cases':results,'counts':{s:sum(r['status']==s for r in results) for s in ['pass','unmet','error']},
                'limitations':['No Milvus or MongoDB integration is installed or tested.',
                               'Injected I/O failures and subprocess termination are real SQLite operations in temporary databases.',
                               'Most fault cases use explicit hash fixtures to isolate storage behavior; three checks verify actual E5 encoding, token budget, and ingestion measurements.',
                               'Requirements missing by design are reported unmet, not silently counted as successful equivalents.']}
        Path(output).parent.mkdir(exist_ok=True,parents=True)
        Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report['counts']),flush=True)
        return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT/'results.json');parser.add_argument('--no-real-model',action='store_true');parser.add_argument('--fail-on-unmet',action='store_true',help='exit 1 if any engineering contract is unmet')
    args=parser.parse_args();report=run(args.output,not args.no_real_model)
    if args.fail_on_unmet and (report['counts']['unmet'] or report['counts']['error']):raise SystemExit(1)
