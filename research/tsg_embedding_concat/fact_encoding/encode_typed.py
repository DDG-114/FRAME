"""Frozen encoder for dynamic and usual typed-fact caches."""
from pathlib import Path
import hashlib
import os
import time
import numpy as np
import torch
from cache_tools import HERE,read,save,sha,input_text,token_rows,pool_last_l2,row_positions


def manifest(tag,cf,**extra):
    folder=HERE/'caches'/tag;ids,_=row_positions();schema='schema' in tag
    save(folder/'MANIFEST.json',dict(status='complete',encoder='schema' if schema else 'embedding',representation='schema' if schema else 'pretrained',
        model_id='field_bound_numeric' if schema else 'Qwen/Qwen3-Embedding-8B',semantic_type=tag,encoder_frozen=True,LoRA=False,
        L2_normalized=True,TRAIN_centered=False,ids=ids,shape=[18612,4096],vectors_sha256=sha(folder/'vectors.npy'),
        pooling='field_bound_TRAIN_scaled_anchor_L2' if schema else 'last_valid_token_FP32_L2',max_tokens=512,truncated=0,
        corpus_file=cf,corpus_sha256=sha(HERE/cf),rows_file='TEXT_ROWS.npz',rows_sha256=sha(HERE/'TEXT_ROWS.npz'),
        fact_rows_sha256=sha(HERE/('USUAL_ROWS.json' if 'usual' in tag else 'FACT_ROWS.json')),
        schema_stats_sha256=sha(HERE/'SCHEMA_STATS.json'),registration_sha256=sha(HERE/'REGISTRATION.json'),optimizer_updates=0,**extra))


def main():
    plan=read(HERE/'PLAN.json');assert os.environ['PAPER_GPU_UUID']==plan['lanes']['102_1']['uuid']
    from transformers import AutoModel,AutoTokenizer
    from structured import matrix
    cfg=read(HERE/'EXPECTED_MODELS.json')['embedding'];source=Path(plan['source_dynamic_cache']['remote'])
    assert sha(source/'MANIFEST.json')==plan['source_dynamic_cache']['manifest_sha256'] and sha(source/'vectors.npy')==plan['source_dynamic_cache']['vectors_sha256']
    tokenizer=AutoTokenizer.from_pretrained(cfg['path'],local_files_only=True,padding_side='right');tokenizer.pad_token=tokenizer.eos_token
    for suffix,rf,cf in [('dynamic','FACT_ROWS.json','TEXT_TYPED.json'),('usual','USUAL_ROWS.json','TEXT_USUAL.json')]:
        tag='typed_schema_'+suffix;folder=HERE/'caches'/tag;folder.mkdir(parents=True,exist_ok=False)
        vectors,width=matrix(read(HERE/rf),read(HERE/'SCHEMA_STATS.json'));np.save(folder/'vectors.npy',vectors);del vectors
        manifest(tag,cf,new_forward_windows=0,active_feature_width=width)
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(32000*1048576/torch.cuda.get_device_properties(0).total_memory)
    model=AutoModel.from_pretrained(cfg['path'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='eager').eval().requires_grad_(False).to('cuda:0')
    original=read(HERE/'TEXT_CORPUS.json');started=time.time()
    with torch.inference_mode():
        rows=token_rows(tokenizer,[input_text(x['text']) for x in original[:16]])
        batch=tokenizer.pad({'input_ids':rows},padding=True,return_tensors='pt').to('cuda:0')
        replay=pool_last_l2(model(**batch,use_cache=False).last_hidden_state,batch['attention_mask']).cpu().numpy()
        assert np.array_equal(replay,np.load(source/'vectors.npy',mmap_mode='r')[:16]),'Source same-batch replay not bitwise exact'
        save(HERE/'SOURCE_REPLAY.json',dict(status='passed',windows=16,bitwise_exact=True,source_sha256=plan['source_dynamic_cache']['vectors_sha256']))
        for suffix,cf in [('dynamic','TEXT_TYPED.json'),('usual','TEXT_USUAL.json')]:
            tag='typed_embedding_'+suffix;folder=HERE/'caches'/tag;folder.mkdir(parents=True,exist_ok=False)
            corpus=read(HERE/cf);out=np.lib.format.open_memmap(folder/'vectors.npy',mode='w+',dtype='float32',shape=(18612,4096));digest=hashlib.sha256();t0=time.time()
            for start in range(0,len(corpus),16):
                tokens=token_rows(tokenizer,[input_text(x['text']) for x in corpus[start:start+16]]);assert max(map(len,tokens))<=512
                for row in tokens:digest.update(np.asarray([len(row),*row],dtype='<i8').tobytes())
                batch=tokenizer.pad({'input_ids':tokens},padding=True,return_tensors='pt').to('cuda:0')
                raw=pool_last_l2(model(**batch,use_cache=False).last_hidden_state,batch['attention_mask']).cpu().numpy();assert np.isfinite(raw).all()
                if start==0:first=(batch,raw.copy())
                out[start:start+len(raw)]=raw
                if start%160==0 or start+len(raw)==len(corpus):
                    out.flush();save(HERE/'ENCODING_PROGRESS.json',dict(arm=tag,completed=start+len(raw),total=18612,seconds=time.time()-t0,pid=os.getpid(),optimizer_updates=0))
            batch,expected=first;replay=pool_last_l2(model(**batch,use_cache=False).last_hidden_state,batch['attention_mask']).cpu().numpy();assert np.array_equal(replay,expected)
            out.flush();del out
            manifest(tag,cf,new_forward_windows=18612,same_batch_replay_windows=16,same_batch_replay_exact=True,token_input_sha256=digest.hexdigest(),postload_GPU_seconds=time.time()-t0)
    save(HERE/'ENCODING_FINISH.json',dict(status='passed',typed_forward_windows=37224,source_replay_windows=16,modified_replay_windows=32,
        source16_bitwise_exact=True,postload_GPU_seconds=time.time()-started,optimizer_updates=0,qualification_updates=0,formal_updates=0,new_forecasting_weights=0))


if __name__=='__main__':main()
