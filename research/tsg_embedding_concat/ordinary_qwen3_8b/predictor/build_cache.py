"""Frozen pretrained/random Qwen features or a tokenizer-only MLP control input."""
from pathlib import Path
import argparse
import json
import os
import numpy as np
import torch
from data import HERE, sha
from model import masked_token_mean


def main(args):
    if args.representation != 'schema':
        assert 'PAPER_GPU_UUID' in os.environ
        torch.cuda.set_per_process_memory_fraction(args.cap_mib*1048576/torch.cuda.get_device_properties(0).total_memory)
    config=json.loads((args.model/'config.json').read_text(encoding='utf-8'))
    assert config['hidden_size']==4096 and config['model_type']=='qwen3'
    assert 'embedding' not in str(args.model).lower()
    from transformers import AutoModel,AutoTokenizer,AutoConfig
    corpus_path=HERE/args.corpus;rows_path=HERE/args.rows
    corpus=json.loads(corpus_path.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,padding_side='right')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=None
    torch.set_num_threads(1)
    if args.representation=='pretrained':
        model=AutoModel.from_pretrained(args.model,torch_dtype=torch.bfloat16,local_files_only=True,
                                       attn_implementation='eager').eval().requires_grad_(False).to('cuda:0')
    elif args.representation=='random':
        torch.manual_seed(2027)
        model=AutoModel.from_config(AutoConfig.from_pretrained(args.model,local_files_only=True),
                                    torch_dtype=torch.bfloat16,attn_implementation='eager')
        model=model.eval().requires_grad_(False).to('cuda:0')
    vectors=np.lib.format.open_memmap(args.output/'vectors.npy',mode='w+',dtype='float32',shape=(len(corpus),4096))
    missing=list(range(len(corpus)));reused=0
    if args.reuse_from:
        m=json.loads((args.reuse_from/'MANIFEST.json').read_text(encoding='utf-8'))
        assert args.representation=='pretrained' and m.get('representation','pretrained')=='pretrained'
        assert m['vectors_sha256']==sha(args.reuse_from/'vectors.npy')
        assert m['model_config_sha256']==sha(args.model/'config.json')
        old=np.load(args.reuse_from/'vectors.npy',mmap_mode='r');lookup={key:i for i,key in enumerate(m['ids'])}
        missing=[]
        for i,item in enumerate(corpus):
            if item['id'] in lookup:
                vectors[i]=old[lookup[item['id']]];reused+=1
            else:missing.append(i)
    with torch.no_grad():
        for start in range(0,len(missing),args.batch):
            positions=missing[start:start+args.batch]
            texts=[corpus[i]['text'] for i in positions]
            tokens=tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
            if model is None:
                for j in range(len(texts)):
                    ids=tokens['input_ids'][j][tokens['attention_mask'][j].bool()].numpy()
                    vectors[positions[j]]=np.bincount(ids%4096,minlength=4096).astype(np.float32)/max(len(ids),1)
            else:
                tokens=tokens.to('cuda:0')
                hidden=model(**tokens,use_cache=False).last_hidden_state.float()
                vectors[positions]=masked_token_mean(hidden,tokens['attention_mask']).cpu().numpy()
            vectors.flush()
            print(json.dumps({'encoded':start+len(texts),'total_new':len(missing),'reused':reused,'representation':args.representation}),flush=True)
    weights={p.name:sha(p) for p in sorted(args.model.glob('*.safetensors'))} if args.representation=='pretrained' else {}
    if model is not None:assert not any(p.requires_grad for p in model.parameters())
    m=dict(status='complete',model_id='schema_token_counts' if model is None else 'Qwen/Qwen3-8B',
           representation=args.representation,condition=args.condition,omit_availability=args.omit_availability,
           corpus_file=args.corpus,rows_file=args.rows,corpus_sha256=sha(corpus_path),rows_sha256=sha(rows_path),
           model_config_sha256=sha(args.model/'config.json'),model_weight_sha256=weights,
           random_encoder_seed=2027 if args.representation=='random' else None,
           random_encoder_initialization='AutoModel.from_config; public weights not loaded' if args.representation=='random' else None,
           ids=[x['id'] for x in corpus],vectors_sha256=sha(args.output/'vectors.npy'),shape=[len(corpus),4096],
           pooling='token frequency hashed by tokenID modulo4096; trainable two-layer schema MLP is inside predictor' if model is None else 'nonpadding final-token-state mean',
           max_tokens=512,L2_normalized=False,TRAIN_centered=False,encoder_frozen=True,LoRA=False,optimizer_updates=0,
           GPU_UUID=os.environ.get('PAPER_GPU_UUID'),tokenizer_sha256=sha(args.model/'tokenizer.json'))
    m['unchanged_text_vectors_reused_exactly']=reused
    m['reused_manifest_sha256']=sha(args.reuse_from/'MANIFEST.json') if args.reuse_from else None
    (args.output/'MANIFEST.json').write_text(json.dumps(m,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--representation',choices=('pretrained','random','schema'),default='pretrained')
    p.add_argument('--condition',choices=('complete','remove','delay24h'),default='complete')
    p.add_argument('--omit-availability',action='store_true')
    p.add_argument('--corpus',default='TEXT_CORPUS.json');p.add_argument('--rows',default='TEXT_ROWS.npz')
    p.add_argument('--reuse-from',type=Path)
    p.add_argument('--cap-mib',type=int,default=32000);main(p.parse_args())
