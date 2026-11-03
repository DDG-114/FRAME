"""Field-aware, information-preserving numeric control for the same facts."""
from fractions import Fraction
import math
import numpy as np
from cache_tools import row_positions
from data import TASKS,calendars
from typed_facts import FACTS,CHANGES

META=('day','minute','horizon','age','n24','nprev24','lead_first','lead_last','cadence')


def standard(values):
    values=list(map(int,values));n=len(values)
    if not n:return dict(mean=0.,scale=1.,count=0)
    mean=Fraction(sum(values),n);variance=Fraction(sum(x*x for x in values),n)-mean*mean
    return dict(mean=float(mean),scale=math.sqrt(float(variance)) if variance else 1.,count=n)


def fit(data,records):
    _,mapping=row_positions();stats={};assets=[]
    for task,positions in mapping.items():
        selected=[records[positions[i]] for i in data.splits[task]['train']]
        stats[task]=dict(facts=[standard([r['values'][j] for r in selected if r['values'][j] is not None]) for j in range(len(FACTS))],
            metadata={k:standard([r['metadata'][k] for r in selected]) for k in META})
        assets.extend(task+'/'+asset for asset in sorted({r['metadata']['asset'] for r in selected}))
    return dict(tasks=stats,asset_vocabulary=assets,fit_rows={t:list(map(int,data.splits[t]['train'])) for t in mapping},
        source='TRAIN typed input facts only; no labels/VAL fitting',normalization='Per-task TRAIN mean/std; constant-feature scale1; final L2 with an anchor1 keeps ratios recoverable',
        fields=list(FACTS),metadata_fields=list(META),allocated_width=4096)


def vector(record,stats):
    m=record['metadata'];task=record['task'];s=stats['tasks'][task]
    v=[1.]+[float(task==t) for t in TASKS]+[float(m['cadence']==c) for c in [60,10,30]]
    identity=task+'/'+m['asset'];assert identity in stats['asset_vocabulary']
    v += [float(identity==a) for a in stats['asset_vocabulary']]
    metadata_first=len(v)
    v += [(m[k]-s['metadata'][k]['mean'])/s['metadata'][k]['scale'] for k in META]
    v += list(map(float,m['availability']))
    v += list(map(float,m['future_present']))+[0.]*(6-len(m['future_present']))
    origin=np.datetime64(m['day']*1440+m['minute'],'m');cal,mask=calendars(np.asarray([origin]),weekly=task!='wind')
    v += cal[0].astype(float).tolist()+mask[0].astype(float).tolist()
    first=len(v)
    v += [0. if x is None else (x-f['mean'])/f['scale'] for x,f in zip(record['values'],s['facts'])]
    v += [float(x is not None) for x in record['values']]
    for key in FACTS:
        if key in CHANGES:
            x=record['values'][FACTS.index(key)];v += [float(x is not None and x<0),float(x==0),float(x is not None and x>0)]
    norm=math.sqrt(math.fsum(x*x for x in v));out=np.zeros(4096,np.float32);out[:len(v)]=np.asarray(v,np.float64)/norm
    assert np.isfinite(out).all() and abs(float(np.linalg.norm(out))-1)<1e-6
    # Quantized text values must be recoverable exactly after the float32 interface.
    for j,(x,f) in enumerate(zip(record['values'],s['facts'])):
        if x is not None:assert int(np.rint(float(out[first+j])/float(out[0])*f['scale']+f['mean']))==x,(task,FACTS[j],x)
    for j,k in enumerate(META):
        f=s['metadata'][k];assert int(np.rint(float(out[metadata_first+j])/float(out[0])*f['scale']+f['mean']))==m[k],(task,k)
    return out,len(v)


def matrix(records,stats):
    out=np.empty((len(records),4096),np.float32);width=None
    for i,r in enumerate(records):
        out[i],n=vector(r,stats)
        assert width is None or width==n;width=n
    return out,width
