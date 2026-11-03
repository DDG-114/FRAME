"""Independent CPU checks of every saved candidate, sampler and selection."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
from data import HERE,SOURCE,TASKS,BATCH,Dataset,protocol,sha
from model import Config,PaperForecaster


def audit(run, source=SOURCE):
    T,E=protocol(source)
    contract=json.loads((HERE/'INPUT_CONTRACT.json').read_text(encoding='utf-8'))
    data=Dataset(source,context_stats=contract['context_stats'])
    assert data.kappa==contract['kappa']
    config=json.loads((run/'CONFIGURATION.json').read_text(encoding='utf-8'))
    history=json.loads((run/'HISTORY.json').read_text(encoding='utf-8'))
    updates=[json.loads(line) for line in (run/'TRAIN_UPDATES.jsonl').read_text(encoding='utf-8').splitlines()]
    steps=config['steps']
    assert steps==(3 if config['qualification'] else 2560)
    assert [r['step'] for r in updates]==list(range(1,steps+1))
    expected=[0,3] if config['qualification'] else list(range(0,2561,128))
    assert [r['step'] for r in history]==expected
    torch.manual_seed(2027)
    model=PaperForecaster(Config(mode=config['mode']))
    initial_state={n:v.clone() for n,v in model.state_dict().items()}
    names=[n for n,_ in model.named_parameters()]
    counts={n:0 for n in names}
    streams={t:T.NativeBatchStream(np.asarray(data.splits[t]['train']),BATCH[t],seed=2027,task=t) for t in TASKS}
    frames={0:({n:0 for n in names},{t:dict(order=s.order.copy(),position=s.position,rng=s.rng.bit_generator.state.copy(),digest=s.digest.hexdigest()) for t,s in streams.items()})}
    for row in updates:
        assert row['learning_rate']==3e-4 and np.isfinite(row['preclip_norm'])
        assert abs(row['objective']-sum(row['task_objectives'].values())/4)<1e-12
        for t,s in streams.items():s.next();assert s.digest.hexdigest()==row['sampler_digests'][t]
        assert set(row['optimizer_grad_none'])<=set(names)
        for n in names:counts[n]+=n not in row['optimizer_grad_none']
        if row['step'] in expected:
            frames[row['step']]=(counts.copy(),{t:dict(order=s.order.copy(),position=s.position,rng=s.rng.bit_generator.state.copy(),digest=s.digest.hexdigest()) for t,s in streams.items()})
    verified=[]
    for row in history:
        step=row['step'];checkpoint=run/'checkpoints'/f'step_{step:04d}.pt'
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert cp['step']==step and cp['seed']==2027
        assert cp['model_config']==config['model'] and not cp['EMA'] and not cp['language_LoRA']
        if step==0:
            assert all(torch.equal(v,initial_state[n]) for n,v in cp['state_dict'].items()),'Fresh2027 initialization differs'
        model.load_state_dict(cp['state_dict'],strict=True)
        assert all(torch.isfinite(v).all() for v in cp['state_dict'].values())
        assert cp['optimizer_parameter_names']==names
        assert cp['optimizer_ages']==cp['optimizer_grad_update_counts']==frames[step][0]
        assert cp['input_contract_sha256']==sha(HERE/'INPUT_CONTRACT.json')
        for task,s in cp['sampler_states'].items():
            expected_state=frames[step][1][task]
            assert np.array_equal(s['order'],expected_state['order']) and s['position']==expected_state['position']
            assert s['rng']==expected_state['rng'] and s['digest']==expected_state['digest']
        groups=cp['optimizer']['param_groups'];assert len(groups)==1
        group=groups[0]
        assert group['lr']==3e-4 and group['betas']==(.9,.999) and group['weight_decay']==1e-4
        for name,index in zip(names,group['params']):
            state=cp['optimizer']['state'].get(index,{})
            assert int(state.get('step',0))==frames[step][0][name]
            for v in state.values():
                if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
        raw_count=0
        for split in ('train','val'):
            for task in TASKS:
                path=run/'candidates'/f'step_{step:04d}'/(task+'_'+split+'_predictions.npz')
                rows=np.asarray(data.splits[task][split]);a=data.arrays[task]
                raw,metadata=E.load_predictions(path)
                for k in ('sample_id','origin_time','target_raw','target_mask','valid_time','mase_scale'):
                    assert np.array_equal(raw[k],a[k][rows]),(task,split,step,k)
                assert np.isfinite(raw['prediction_raw']).all()
                if split=='val':
                    _,report=E.validation_asset_macro_mase(raw)
                else:
                    groups=E.evaluate_arrays(raw)['groups']
                    assert len(groups)==1
                    report=groups[0]
                for metric in ('nmae','rmse','smape','mase'):
                    assert np.isclose(report['asset_macro'][metric],row[split][task][metric],atol=1e-14,rtol=0),(task,split,step,metric)
                raw_count+=1
        assert np.isclose(row['score'],sum(row['val'][t]['nmae'] for t in TASKS)/4,atol=1e-14,rtol=0)
        verified.append(dict(step=step,raw_files=raw_count,checkpoint_sha256=sha(checkpoint)))
    chosen=min((x for x in history if x['eligible']),key=lambda x:(x['score'],x['step']))
    summary=json.loads((run/'SUMMARY.json').read_text(encoding='utf-8'))
    assert summary['selected']==chosen
    assert sha(run/'SELECTED.pt')==summary['selected_sha256']==sha(run/'checkpoints'/f"step_{chosen['step']:04d}.pt")
    cold=json.loads((run/'COLD_RESTORE.json').read_text(encoding='utf-8'))
    assert cold['status']=='passed' and cold['step']==chosen['step'] and cold['checkpoint_sha256']==sha(run/'SELECTED.pt')
    result=dict(status='all_candidates_independently_verified',mode=config['mode'],formal_updates=0 if config['qualification'] else steps,
        qualification_updates=steps if config['qualification'] else 0,verified=verified,raw_count=sum(x['raw_files'] for x in verified),
        selected=chosen,TEST_evaluated=False,selection='four-task VAL asset-macro NMAE, earliest tie',
        candidate_metric_atol=1e-14,cohort_exact=True,sampler_exact=True,optimizer_ages_exact=True,cold_exact=True)
    (run/'AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(HERE/'plot.py'),str(run)],check=True)
    print(json.dumps({'status':result['status'],'raw_count':result['raw_count'],'selected_step':chosen['step']}))


def audit_condition(run,folder,source=SOURCE):
    _,E=protocol(source)
    result=json.loads((folder/'CONDITION_RESULT.json').read_text(encoding='utf-8'))
    assert result['optimizer_updates']==0 and result['model_parameters_unchanged'] and not result['selection_changed']
    assert result['source_selected_sha256']==sha(run/'SELECTED.pt')
    trained=json.loads((run/'AUDIT.json').read_text(encoding='utf-8'))
    assert trained['status']=='all_candidates_independently_verified'
    assert result['selected_step']==trained['selected']['step']
    contract=json.loads((HERE/'INPUT_CONTRACT.json').read_text(encoding='utf-8'))
    data=Dataset(source,context_stats=contract['context_stats'],condition=result['condition'],
                 omit_availability=result['mode']=='no_availability')
    for split in ('train','val'):
        for task in TASKS:
            raw,_=E.load_predictions(folder/(task+'_'+split+'_predictions.npz'))
            rows=np.asarray(data.splits[task][split]);a=data.arrays[task]
            for key in ('sample_id','origin_time','target_raw','target_mask','valid_time','mase_scale'):
                assert np.array_equal(raw[key],a[key][rows]),(task,split,key)
            report=E.validation_asset_macro_mase(raw)[1] if split=='val' else E.evaluate_arrays(raw)['groups'][0]
            for metric in ('nmae','rmse','smape','mase'):
                assert np.isclose(report['asset_macro'][metric],result['metrics'][split][task][metric],atol=1e-14,rtol=0)
    checked=dict(status='independently_verified',raw_files=8,source_selected_sha256=result['source_selected_sha256'],
                 condition=result['condition'],optimizer_updates=0,selected_step=result['selected_step'],metrics=result['metrics'])
    (folder/'AUDIT.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
    print(json.dumps({'status':checked['status'],'condition':result['condition']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--condition-folder',type=Path)
    a=p.parse_args()
    if a.condition_folder:audit_condition(a.run,a.condition_folder,a.source)
    else:audit(a.run,a.source)
