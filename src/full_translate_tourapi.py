"""Research-only Korean->EN/JA full-description baseline, using local NLLB.

NLLB-200 is intended for sentence translation. Split into sentences (and
token-safe chunks for unusually long sentences), translate each segment, then
reassemble. Cache every translated segment for resuming interrupted runs.
"""
import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/processed/tourapi_standardized.csv'
OUT=ROOT/'results/tourapi_full_translation'
DEST=ROOT/'data/processed/tourapi_standardized_full_translation_experiment.csv'
MODEL='facebook/nllb-200-distilled-600M'
TARGETS={'en':'eng_Latn','ja':'jpn_Jpan'}

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)

def segment(text,tokenizer,max_tokens=220):
    # Remove only a non-semantic CSS artifact observed in one API description.
    text=re.sub(r'\.text202503\s*\{\s*COLOR\s*:\s*#FF0000\s*;\s*\}', '',text,flags=re.I)
    text=' '.join(text.split())
    sentences=[s.strip() for s in re.split(r'(?<=[.!?。！？])\s+',text) if s.strip()]
    def n(t):return len(tokenizer(t,add_special_tokens=True,truncation=False)['input_ids'])
    parts=[]
    for s in sentences:
        if n(s)<=max_tokens:
            parts.append(s);continue
        words=s.split(); chunk=[]
        for word in words:
            candidate=' '.join(chunk+[word])
            if chunk and n(candidate)>max_tokens:
                parts.append(' '.join(chunk));chunk=[]
            chunk.append(word)
        if chunk:parts.append(' '.join(chunk))
    assert ' '.join(parts)==text, 'source content lost in segmentation'
    assert all(n(s)<=max_tokens for s in parts)
    return parts

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    p=argparse.ArgumentParser()
    p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--device',choices=('mps','cpu'),default=None)
    p.add_argument('--model-path',default=MODEL,help='Hugging Face model ID or a local model snapshot directory')
    p.add_argument('--revision',default=None,help='Optional Hugging Face model revision')
    p.add_argument('--local-files-only',action='store_true',help='Use only an already downloaded model')
    args=p.parse_args()
    args.device=args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model_options={'revision':args.revision,'local_files_only':args.local_files_only}
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    docs=read(SOURCE);ko=[r for r in docs if r['language']=='ko']
    assert len(docs)==300 and len(ko)==100
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print('Preparing model',MODEL,args.device,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,src_lang='kor_Hang',**model_options)
    segments=[]
    for d in ko:
        for i,text in enumerate(segment(d['description'],tokenizer),1):
            count=len(tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'])
            segments.append(dict(place_id=d['place_id'],segment_index=i,source_text=text,source_tokens=count,source_sha256=hashlib.sha256(text.encode()).hexdigest()))
    assert len(segments)>=100 and len({(x['place_id'],x['segment_index']) for x in segments})==len(segments)
    write(OUT/'source_segments.csv',segments)
    cache_path=OUT/'translation_cache.jsonl';cache={}
    if cache_path.exists():
        for line in cache_path.open(encoding='utf-8'):
            try:
                rec=json.loads(line);rec.setdefault('decoding','GREEDY_OR_INITIAL_RETRY');key=(rec['place_id'],rec['segment_index'],rec['language'],rec['source_sha256'])
                if rec['translation'].strip() and not rec['output_capped']:
                    cache[key]=rec
            except (ValueError,KeyError,TypeError):continue
    needed={(s['place_id'],s['segment_index'],lang,s['source_sha256']) for s in segments for lang in TARGETS}
    model=None
    if needed-set(cache):
        model=AutoModelForSeq2SeqLM.from_pretrained(args.model_path,dtype=torch.float16 if args.device=='mps' else torch.float32,**model_options).to(args.device).eval()
    print('Source segments',len(segments),'translation tasks',len(segments)*2,'cached',len(cache),flush=True)
    def translate_batch(batch,lang,max_new_tokens=None,recovery=False):
        assert model is not None
        texts=[r['source_text'] for r in batch]
        inputs=tokenizer(texts,return_tensors='pt',padding=True,truncation=False).to(args.device)
        limit=max_new_tokens or min(512,max(48,max(r['source_tokens'] for r in batch)*2+32))
        forced=tokenizer.convert_tokens_to_ids(TARGETS[lang]);assert forced!=tokenizer.unk_token_id
        options=dict(forced_bos_token_id=forced,max_new_tokens=limit,num_beams=1,do_sample=False,use_cache=True)
        # A few specialized architectural terms can trigger repetition loops.
        if recovery: options.update(no_repeat_ngram_size=3,repetition_penalty=1.15,num_beams=2)
        with torch.inference_mode():
            generated=model.generate(**inputs,**options)
        raw=generated.detach().cpu().tolist()
        decoded=tokenizer.batch_decode(raw,skip_special_tokens=True)
        results=[]
        for row,tokens,text in zip(batch,raw,decoded):
            output_ids=tokens[1:]
            # Require an EOS before the length cap; model can force EOS at cap.
            eos=tokenizer.eos_token_id
            first_end=output_ids.index(eos)+1 if eos in output_ids else len(output_ids)
            capped=eos not in output_ids or first_end>=limit
            rec=dict(**row,language=lang,translation=text.strip(),output_tokens=first_end,output_capped=capped,decoding='RECOVERY' if recovery else 'GREEDY')
            results.append(rec)
        return results
    with cache_path.open('a',encoding='utf-8') as f:
        for lang in TARGETS:
            todo=[r for r in segments if (r['place_id'],r['segment_index'],lang,r['source_sha256']) not in cache]
            todo.sort(key=lambda r:r['source_tokens'])
            for i in range(0,len(todo),args.batch_size):
                batch=todo[i:i+args.batch_size]
                results=translate_batch(batch,lang)
                for row,rec in zip(batch,results):
                    if rec['output_capped'] or not rec['translation']:
                        rec=translate_batch([row],lang,max_new_tokens=256,recovery=True)[0]
                    if rec['output_capped'] or not rec['translation']:
                        raise RuntimeError(f'Translation incomplete: {row["place_id"]}/{row["segment_index"]}/{lang}')
                    cache[rec['place_id'],rec['segment_index'],lang,rec['source_sha256']]=rec
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                f.flush()
                print(f'{lang}: {min(i+args.batch_size,len(todo))}/{len(todo)} new segments',flush=True)
    ordered=[];by_doc=defaultdict(list)
    for lang in TARGETS:
        for s in segments:
            rec=cache[s['place_id'],s['segment_index'],lang,s['source_sha256']]
            rec.setdefault('decoding','GREEDY_OR_INITIAL_RETRY');ordered.append(rec);by_doc[rec['place_id'],lang].append(rec)
    assert len(ordered)==len(segments)*2
    write(OUT/'translation_segments.csv',ordered)
    final=[];checks=[]
    ko_index={r['place_id']:r for r in ko}
    for d in docs:
        row=dict(d)
        if d['language'] in TARGETS:
            pieces= sorted(by_doc[d['place_id'],d['language']],key=lambda r:r['segment_index'])
            row['description']=' '.join(x['translation'] for x in pieces)
            assert row['description'].strip()
            checks.append(dict(place_id=d['place_id'],language=d['language'],title=d['title'],segments=len(pieces),korean_chars=len(ko_index[d['place_id']]['description']),translated_chars=len(row['description']),original_foreign_chars=len(d['description']),korean_source_chars_ratio=round(len(row['description'])/len(ko_index[d['place_id']]['description']),4),output_capped=sum(x['output_capped'] for x in pieces),contains_hangul='YES' if re.search(r'[가-힣]',row['description']) else 'NO',contains_japanese_script='YES' if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]',row['description']) else 'NO',verification='UNREVIEWED_MACHINE_TRANSLATION'))
        final.append(row)
    assert Counter(r['language'] for r in final)==Counter(ko=100,en=100,ja=100)
    assert sum(a['description']!=b['description'] for a,b in zip(docs,final))==200
    assert all(a[k]==b[k] for a,b in zip(docs,final) for k in ('source','place_id','language','title'))
    write(DEST,final);write(OUT/'translation_quality_checks.csv',checks)
    (OUT/'translation_metadata.json').write_text(json.dumps(dict(model=MODEL,model_path=args.model_path,model_revision=args.revision,source_sha256=source_hash,segment_count=len(segments),translated_documents=200,translation_tasks=len(ordered),source_language='kor_Hang',targets=TARGETS,batch_size=args.batch_size,device=args.device,num_beams_primary=1,decoding_counts=dict(Counter(r['decoding'] for r in ordered)),recovery='2 beams, repetition penalty and no repeated trigrams for failed segments',do_sample=False,last_run_elapsed_seconds=round(time.monotonic()-start,2),license='CC-BY-NC (research only)',limitation='NLLB is a research sentence-level model, not a certified translation. Segment outputs concatenated; no human quality verification.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Full translation dataset saved',DEST,flush=True)

if __name__=='__main__':main()
