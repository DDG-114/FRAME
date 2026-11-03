"""Prepared full-budget four-task trainer; no old trained weights are loaded."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import copy
import json
import os
import random
import shutil
import time
import numpy as np
import torch
from data import HERE, SOURCE, TASKS, BATCH, Dataset, protocol, sha
from model import Config, PaperForecaster, objective, MODES


def save(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(path)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])


@contextmanager
def evaluation_mode(model):
    state, training=rng_state(),model.training
    try:
        model.eval()
        with torch.no_grad():yield
    finally:
        restore_rng(state)
        model.train(training)


class SemanticCache:
    def __init__(self, path):
        self.path=Path(path)
        self.manifest=json.loads((self.path/'MANIFEST.json').read_text(encoding='utf-8'))
        m=self.manifest
        assert m['status']=='complete' and m['model_id'] in ('Qwen/Qwen3-8B', 'schema_token_counts')
        assert m['encoder_frozen'] and not m['LoRA'] and not m['L2_normalized'] and not m['TRAIN_centered']
        assert m['max_tokens']==512 and m['corpus_sha256']==sha(HERE/m.get('corpus_file','TEXT_CORPUS.json'))
        assert m['vectors_sha256']==sha(self.path/'vectors.npy')
        self.vectors=np.load(self.path/'vectors.npy',mmap_mode='r')
        assert self.vectors.shape==(len(m['ids']),4096)
        lookup={s:i for i,s in enumerate(m['ids'])}
        with np.load(HERE/m.get('rows_file','TEXT_ROWS.npz'),allow_pickle=False) as z:
            self.positions={t:np.asarray([lookup[x] for x in z[t]],np.int64) for t in TASKS}

    def get(self, task, rows):
        return self.vectors[self.positions[task][rows]].copy()


def evaluate(model, dataset, cache, evaluator, split, output, step, device, batch_size=32):
    metrics={}
    with evaluation_mode(model):
        for task in TASKS:
            a=dataset.arrays[task]
            rows=np.asarray(dataset.splits[task][split],np.int64)
            parts=[]
            for start in range(0,len(rows),batch_size):
                rr=rows[start:start+batch_size]
                batch=dataset.batch(task,rr,cache.get(task,rr),device)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                    parts.append(model(batch)['prediction'].cpu().numpy())
            sub={k:v[rows] for k,v in a.items()}
            raw=evaluator.original_units(np.concatenate(parts),sub)
            exchange=evaluator.exchange_arrays(sub,raw)
            if split=='val':
                _,report=evaluator.validation_asset_macro_mase(exchange)
            else:
                groups=evaluator.evaluate_arrays(exchange)['groups']
                assert len(groups)==1
                report=groups[0]
            metrics[task]={k:float(report['asset_macro'][k]) for k in ('nmae','rmse','smape','mase')}
            evaluator.save_predictions(output/(task+'_'+split+'_predictions.npz'),sub,raw,
                metadata=dict(method='PaperAligned_'+model.config.mode,seed=2027,step=step,split=split,
                              weight_state='live',complete_split=True))
    return metrics


def checkpoint(path, step, model, optimizer, streams, grad_counts, history, cache):
    names=[n for n,_ in model.named_parameters()]
    ages={n:int(optimizer.state.get(p,{}).get('step',0)) for n,p in model.named_parameters()}
    assert ages==grad_counts
    payload=dict(step=step,seed=2027,model_config=model.configuration(),
        state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
        optimizer_parameter_names=names,optimizer_ages=ages,optimizer_grad_update_counts=grad_counts.copy(),
        sampler_states={t:dict(order=s.order.copy(),position=s.position,rng=copy.deepcopy(s.rng.bit_generator.state),digest=s.digest.hexdigest()) for t,s in streams.items()},
        rng=rng_state(),selection_history=history,semantic_manifest_sha256=sha(cache.path/'MANIFEST.json'),
        input_contract_sha256=sha(HERE/'INPUT_CONTRACT.json'),prepared_science_sha256=sha(HERE/'PREPARATION_MANIFEST.json'),
        EMA=False,language_LoRA=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.part')
    torch.save(payload,tmp)
    tmp.replace(path)


def runtime(args, mode):
    if 'PAPER_GPU_UUID' not in os.environ:
        raise RuntimeError('Run through gpu_entry.py; prepared runs require an exclusive UUID lease')
    torch.cuda.set_per_process_memory_fraction(args.cap_mib*1048576/torch.cuda.get_device_properties(0).total_memory)
    T,E=protocol(args.source)
    T.configure_reproducibility(2027,True)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    prepared=json.loads((HERE/'PREPARATION_MANIFEST.json').read_text(encoding='utf-8'))
    for name,h in prepared['files'].items():assert sha(HERE/name)==h,name
    contract=json.loads((HERE/'INPUT_CONTRACT.json').read_text(encoding='utf-8'))
    assert sha(args.source/'bundle/SPLIT_ROWS.json')==contract['split_sha256']
    for task in TASKS:assert sha(args.source/'bundle/native'/(task+'_trainval.npz'))==contract['native_sha256'][task]
    for name,h in contract['protocol_sha256'].items():assert sha(args.source/'protocol'/name)==h,name
    data=Dataset(args.source,context_stats=contract['context_stats'],
                 condition=getattr(args,'context_condition','complete'),omit_availability=mode=='no_availability')
    assert data.kappa==contract['kappa']
    cache=SemanticCache(args.cache)
    expected_representation='schema' if mode=='mlp_schema' else 'random' if mode=='frozen_random' else 'pretrained'
    assert cache.manifest.get('representation','pretrained')==expected_representation
    assert cache.manifest.get('condition','complete')==getattr(args,'context_condition','complete')
    assert cache.manifest.get('omit_availability',False)==(mode=='no_availability')
    model=PaperForecaster(Config(mode=mode)).to('cuda:0')
    return T,E,data,cache,model


def main(args):
    if args.evaluate_selected:
        evaluate_selected(args)
        return
    if args.verify_run:
        verify(args)
        return
    qualification=args.qualification_run or HERE/'qualification'/args.mode
    if not args.qualify:
        q=json.loads((qualification/'AUDIT.json').read_text(encoding='utf-8'))
        assert q['status']=='all_candidates_independently_verified' and q['qualification_updates']==3 and q['mode']==args.mode
    T,E,data,cache,model=runtime(args,args.mode)
    args.output.mkdir(parents=True,exist_ok=False)
    steps=3 if args.qualify else 2560
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
    streams={t:T.NativeBatchStream(np.asarray(data.splits[t]['train']),BATCH[t],seed=2027,task=t) for t in TASKS}
    counts={n:0 for n,_ in model.named_parameters()}
    history=[]
    save(args.output/'CONFIGURATION.json',dict(mode=args.mode,seed=2027,steps=steps,qualification=args.qualify,
        input_contract_sha256=sha(HERE/'INPUT_CONTRACT.json'),semantic_manifest_sha256=sha(args.cache/'MANIFEST.json'),
        model=model.configuration(),optimizer='AdamW(.9,.999),3e-4,WD1e-4,clip1,constant',
        loss='task-mean-absolute-scale maskedHuber plus1e-4 pre-strength candidateenergy',
        selection='complete four-task VAL asset-macro NMAE mean, earliest tie',EMA=False,LoRA=False))
    best=None
    log=(args.output/'TRAIN_UPDATES.jsonl').open('x',encoding='utf-8')
    started=time.time()
    for step in range(steps+1):
        if step:
            t0=time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses={}
            for task in TASKS:
                rows=streams[task].next()
                losses[task]=0.
                for start in range(0,len(rows),4):
                    rr=rows[start:start+4]
                    batch=data.batch(task,rr,cache.get(task,rr),'cuda:0')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        result=model(batch)
                        loss,_,_=objective(result,batch['target'],batch['target_mask'],batch['target_scale'],data.kappa[task])
                        weight=len(rr)/len(rows)
                        (loss*weight/4).backward()
                    losses[task]+=float(loss.detach())*weight
            none=[]
            for name,p in model.named_parameters():
                if p.grad is None:none.append(name)
                else:
                    assert torch.isfinite(p.grad).all(),name
                    counts[name]+=1
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            optimizer.step()
            row=dict(step=step,task_objectives=losses,objective=sum(losses.values())/4,preclip_norm=norm,
                optimizer_grad_none=none,learning_rate=3e-4,sampler_digests={t:s.digest.hexdigest() for t,s in streams.items()},seconds=time.time()-t0)
            log.write(json.dumps(row)+'\n')
            log.flush()
        if step==0 or step%128==0 or (args.qualify and step==3):
            out=args.output/'candidates'/f'step_{step:04d}'
            out.mkdir(parents=True)
            train=evaluate(model,data,cache,E,'train',out,step,'cuda:0')
            val=evaluate(model,data,cache,E,'val',out,step,'cuda:0')
            score=sum(val[t]['nmae'] for t in TASKS)/4
            entry=dict(step=step,score=score,mean_mase=sum(val[t]['mase'] for t in TASKS)/4,train=train,val=val,eligible=step>0)
            history.append(entry)
            path=args.output/'checkpoints'/f'step_{step:04d}.pt'
            checkpoint(path,step,model,optimizer,streams,counts,history,cache)
            if step==0 and not args.qualify:
                initial=torch.load(qualification/'checkpoints/step_0000.pt',map_location='cpu',weights_only=False)
                assert initial['semantic_manifest_sha256']==sha(cache.path/'MANIFEST.json')
                assert initial['input_contract_sha256']==sha(HERE/'INPUT_CONTRACT.json')
                assert not initial['optimizer']['state'] and all(v==0 for v in initial['optimizer_ages'].values())
                assert all(torch.equal(v.detach().cpu(),initial['state_dict'][n]) for n,v in model.state_dict().items())
                for task in TASKS:
                    for split in ('train','val'):
                        name=task+'_'+split+'_predictions.npz'
                        with np.load(out/name) as a,np.load(qualification/'candidates/step_0000'/name) as b:
                            assert np.array_equal(a['prediction_raw'],b['prediction_raw']),(task,split,'fresh0')
                save(args.output/'FRESH_INITIALIZATION_CHECK.json',dict(status='passed',qualification_weights_discarded=True,
                    initial_model_and_complete_TRAIN_VAL_predictions_exact=True,optimizer_empty=True,extra_optimizer_updates=0))
            if step>0 and (best is None or score<best['score']):best=entry
            save(args.output/'HISTORY.json',history)
        if step%8==0 or step==steps:
            save(args.output/'PROGRESS.json',dict(status='trained_pending_independent_check' if step==steps else 'training',
                step=step,total_steps=steps,best_step=best['step'] if best else None,elapsed_seconds=time.time()-started))
    log.close()
    selected=args.output/'checkpoints'/f"step_{best['step']:04d}.pt"
    shutil.copy2(selected,args.output/'SELECTED.pt')
    save(args.output/'SUMMARY.json',dict(status='training_complete_audit_pending',steps=steps,selected=best,
        selected_sha256=sha(selected),selected_weight_state='live',TRAIN_VAL_only=True,TEST_evaluated=False))


def verify(args):
    run=args.verify_run
    summary=json.loads((run/'SUMMARY.json').read_text(encoding='utf-8'))
    payload=torch.load(run/'SELECTED.pt',map_location='cpu',weights_only=False)
    T,E,data,cache,model=runtime(args,payload['model_config']['mode'])
    assert payload['semantic_manifest_sha256']==sha(args.cache/'MANIFEST.json')
    model.load_state_dict(payload['state_dict'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)
    optimizer.load_state_dict(payload['optimizer'])
    for name,p in model.named_parameters():
        state=optimizer.state.get(p,{})
        assert int(state.get('step',0))==payload['optimizer_ages'][name]
        for value in state.values():
            if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
    out=run/'independent_cold'
    out.mkdir(exist_ok=False)
    for split in ('train','val'):
        evaluate(model,data,cache,E,split,out,payload['step'],'cuda:0')
        for task in TASKS:
            name=task+'_'+split+'_predictions.npz'
            reference=run/'candidates'/f"step_{payload['step']:04d}"/name
            with np.load(out/name) as x,np.load(reference) as y:
                assert np.array_equal(x['prediction_raw'],y['prediction_raw']),(task,split)
    save(run/'COLD_RESTORE.json',dict(status='passed',step=payload['step'],checkpoint_sha256=sha(run/'SELECTED.pt'),
        complete_four_task_TRAIN_VAL_predictions_exact=True,optimizer_ages_exact=True,extra_optimizer_updates=0))


def evaluate_selected(args):
    run=args.evaluate_selected
    payload=torch.load(run/'SELECTED.pt',map_location='cpu',weights_only=False)
    _,E,data,cache,model=runtime(args,payload['model_config']['mode'])
    model.load_state_dict(payload['state_dict'],strict=True)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    args.output.mkdir(parents=True,exist_ok=False)
    result={split:evaluate(model,data,cache,E,split,args.output,payload['step'],'cuda:0') for split in ('train','val')}
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
    save(args.output/'CONDITION_RESULT.json',dict(status='complete',mode=model.config.mode,
        condition=args.context_condition,selected_step=payload['step'],source_selected_sha256=sha(run/'SELECTED.pt'),
        semantic_manifest_sha256=sha(args.cache/'MANIFEST.json'),metrics=result,
        optimizer_updates=0,model_parameters_unchanged=True,selection_changed=False,TEST_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=MODES,default='dual')
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--qualify',action='store_true')
    p.add_argument('--qualification-run',type=Path)
    p.add_argument('--verify-run',type=Path)
    p.add_argument('--evaluate-selected',type=Path)
    p.add_argument('--context-condition',choices=('complete','remove','delay24h'),default='complete')
    p.add_argument('--cap-mib',type=int,default=32000)
    args=p.parse_args()
    if not args.output and not args.verify_run:p.error('--output or --verify-run is required')
    main(args)
