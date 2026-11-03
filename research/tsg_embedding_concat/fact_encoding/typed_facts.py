"""Forecast-origin facts from available history and forecasts."""
from collections import defaultdict
from fractions import Fraction
import math
import numpy as np
from cache_tools import HERE,read,save,row_positions
from context import separate,calendar_keys

FACTS=('y_now_z','y_mean24_z','y_peak24_z','y_d1h_z','y_d3h_z','y_d24h_z','y_daymean_change_z',
       'load_temp_mean24_z','load_temp_d1h_z','wind_speed_mean24_mps','wind_speed_d1h_mps',
       'wind_relative_dir_deg','wind_external_temp_C','blade_pitch_mean_deg',
       'forecast_solar_mean_z','forecast_solar_peak_z','forecast_wind100_mean_mps','forecast_wind100_ramp_mps')
CHANGES={'y_d1h_z','y_d3h_z','y_d24h_z','y_daymean_change_z','load_temp_d1h_z','wind_speed_d1h_mps','forecast_wind100_ramp_mps'}


def q(value):
    return int(np.rint(float(value)*1000)) if value is not None and np.isfinite(value) else None


def avg(values):
    values=[int(x) for x in values if x is not None]
    return int(round(Fraction(sum(values),len(values)))) if values else None


def window(values,valid,times,origin,minutes=1440,before=0):
    take=valid&(times>origin-np.timedelta64(minutes+before,'m'))&(times<=origin-np.timedelta64(before,'m'))
    return [q(x) for x in values[take]]


def latest(values,valid,times,origin,recent=None):
    rows=np.flatnonzero(valid&(times<=origin))
    if recent is not None:rows=rows[times[rows]>origin-np.timedelta64(recent,'m')]
    if not len(rows):return None,None
    i=rows[-1];return q(values[i]),times[i]


def delta(values,valid,times,origin,minutes):
    now,t=latest(values,valid,times,origin)
    if t is None:return None
    rows=np.flatnonzero((times==t-np.timedelta64(minutes,'m'))&valid)
    return now-q(values[rows[-1]]) if len(rows) else None


def physical(values,stats):
    return np.asarray(values,np.float64)*float(stats['scale'])+float(stats['offset'])


def extract(data):
    _,mapping=row_positions();records=[None]*18612
    for task,positions in mapping.items():
        a=data.arrays[task];m=data.manifests[task]
        for i,p in enumerate(positions):
            x=np.asarray(a['history_values'][i],np.float64);mask=a['history_mask'][i].astype(bool);times=a['history_time'][i];origin=a['origin_time'][i]
            assert np.all(times<=origin)
            assert origin==origin.astype('datetime64[m]')
            values={k:None for k in FACTS};cur,_=latest(x[:,0],mask[:,0],times,origin)
            last=window(x[:,0],mask[:,0],times,origin);previous=window(x[:,0],mask[:,0],times,origin,before=1440)
            values.update(y_now_z=cur,y_mean24_z=avg(last),y_peak24_z=max(last) if last else None)
            for minutes,name in [(60,'y_d1h_z'),(180,'y_d3h_z'),(1440,'y_d24h_z')]:values[name]=delta(x[:,0],mask[:,0],times,origin,minutes)
            values['y_daymean_change_z']=avg(last)-avg(previous) if last and previous else None
            if task=='load':
                # All supplied temperature stations are already on their own TRAIN scales.
                temp=np.asarray([float(avg([q(v) for v,good in zip(row[1:],mm[1:]) if good]))/1000 if mm[1:].any() else 0 for row,mm in zip(x,mask)])
                valid=mask[:,1:].any(1)
                values['load_temp_mean24_z']=avg(window(temp,valid,times,origin));values['load_temp_d1h_z']=delta(temp,valid,times,origin,60)
            if task=='wind':
                names=[v['name'].split('_')[-1] for v in m['history_variables']];stats=m['observed_history_statistics']
                def channel(name):
                    j=names.index(name);return physical(x[:,j],stats[name]),mask[:,j]
                speed,valid=channel('Wspd');values['wind_speed_mean24_mps']=avg(window(speed,valid,times,origin));values['wind_speed_d1h_mps']=delta(speed,valid,times,origin,60)
                direction,valid=channel('Wdir');values['wind_relative_dir_deg']=latest(direction,valid,times,origin,recent=1440)[0]
                temp,valid=channel('Etmp');values['wind_external_temp_C']=avg(window(temp,valid,times,origin))
                parts=[channel(k) for k in ['Pab1','Pab2','Pab3']]
                pitch=np.asarray([float(avg([q(v[j]) for v,good in parts if good[j]]))/1000 if any(good[j] for _,good in parts) else 0 for j in range(len(times))])
                valid=np.any(np.stack([good for _,good in parts]),axis=0);values['blade_pitch_mean_deg']=latest(pitch,valid,times,origin,recent=1440)[0]
            if task=='net_load':
                fv=np.asarray(a['known_future_values'][i],np.float64);fm=a['known_future_mask'][i].astype(bool)
                for j in [4,5]:
                    published=a['known_future_effective_publication_time'][i,:,j]
                    assert np.all(~np.isnat(published[fm[:,j]])) and np.all(published[fm[:,j]]<=origin)
                solar=[q(v) for v in fv[fm[:,4],4]];values['forecast_solar_mean_z']=avg(solar);values['forecast_solar_peak_z']=max(solar) if solar else None
                off,scale=m['future_feature_statistics']['gb_nwp_wind_speed_100m'];ws=fv[:,5]*float(scale)+float(off);wind=[q(v) for v in ws[fm[:,5]]]
                values['forecast_wind100_mean_mps']=avg(wind);values['forecast_wind100_ramp_mps']=wind[-1]-wind[0] if len(wind)>1 else None
            minutes=int(origin.astype('datetime64[m]').astype(np.int64));valid_target=np.flatnonzero(mask[:,0])
            age=int((origin-times[valid_target[-1]])/np.timedelta64(1,'m')) if len(valid_target) else -1
            meta=dict(task=task,asset=str(a['asset_id'][i]),day=minutes//1440,minute=minutes%1440,cadence=int(a['cadence_minutes'][i]),horizon=a['target'].shape[1],age=age,
                n24=len(last),nprev24=len(previous),lead_first=int(a['lead_minutes'][i,0]),lead_last=int(a['lead_minutes'][i,-1]),
                availability=data.availability(task,i).astype(int).tolist(),future_present=a['known_future_mask'][i].any(0).astype(int).tolist())
            records[p]=dict(task=task,row=i,metadata=meta,values=[values[k] for k in FACTS])
    assert all(x is not None for x in records)
    return records


def typical(data,records):
    _,mapping=row_positions();buckets=defaultdict(lambda:[[] for _ in FACTS]);origins=defaultdict(set)
    for task,positions in mapping.items():
        a=data.arrays[task]
        for i in data.splits[task]['train']:
            r=records[positions[i]]
            for key in calendar_keys(task,str(a['asset_id'][i]),a['origin_time'][i]):
                origins[key].add(str(a['origin_time'][i]))
                for j,v in enumerate(r['values']):
                    if v is not None:buckets[key][j].append(v)
    medians={key:[int(sorted(v)[(len(v)-1)//2]) if v else None for v in values] for key,values in buckets.items()}
    usual=[]
    for r in records:
        t,i=r['task'],r['row'];a=data.arrays[t];keys=calendar_keys(t,r['metadata']['asset'],a['origin_time'][i]);values=[]
        for j,v in enumerate(r['values']):
            if v is None:values.append(None);continue
            key=next((k for k in keys if (len(origins[k])>=16 or k[-1]=='asset') and medians[k][j] is not None),None)
            assert key is not None,(t,i,FACTS[j],'NoTRAIN reference')
            values.append(medians[key][j])
        usual.append(dict(r,values=values))
    return usual,dict(method='lower median of quantized TRAIN input facts',minimum_unique_origins=16,
        groups={str(k):dict(origins=sorted(origins[k]),medians=medians[k]) for k in medians})


def number(value,change=False):
    if value is None:return 'unavailable'
    sign='-' if value<0 else '+' if change else '';digits=abs(int(value));s=sign+str(digits//1000)+'.'+str(digits%1000).zfill(3)
    if change:s+='('+('up' if value>0 else 'down' if value<0 else 'flat')+')'
    return s


def texts(original,records):
    output=[]
    for old,r in zip(original,records):
        prefix,_,suffix=separate(old['text']);assert prefix.endswith('observed_operation:')
        common=prefix[:-len('observed_operation:')]
        facts=';'.join(k+'='+number(v,k in CHANGES) for k,v in zip(FACTS,r['values']) if v is not None or k.startswith('y_'))
        m=r['metadata'];text=common+'z=original_TRAIN_standardized;typed_facts:'+facts+';n24='+str(m['n24'])+';nprev24='+str(m['nprev24'])+';lead_minutes='+str(m['lead_first'])+'..'+str(m['lead_last'])+suffix
        output.append(dict(id=old['id'],text=text))
    return output


def main():
    from data import Dataset
    data=Dataset(context_stats=read(HERE/'INPUT_CONTRACT.json')['context_stats']);records=extract(data);usual,fit=typical(data,records);original=read(HERE/'TEXT_CORPUS.json')
    save(HERE/'FACT_ROWS.json',records);save(HERE/'USUAL_ROWS.json',usual);save(HERE/'TRAIN_USUAL_FIT.json',fit)
    save(HERE/'TEXT_TYPED.json',texts(original,records));save(HERE/'TEXT_USUAL.json',texts(original,usual))


if __name__=='__main__':main()
