"""Canonical Chapter-4 metrics from Train-only centers, not a new metric formula."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'code/experiments/benchmark_v1'))
from report_snapshot import standard_metrics


def by_hypothesis(graph, z, family):
    """Return train/validation hypothesis representations without held-out labels."""
    if family not in ('shared','class') or graph.qx is None:
        raise ValueError('Only labeled shared/class query graph expected')
    with torch.no_grad():
        train=[];val=[]
        for c in range(graph.k):
            train.append(graph.represent_train(z,family,c).detach().cpu().numpy())
            val.append(graph.represent_query(z,family,c).detach().cpu().numpy())
    return np.stack(train,axis=1),np.stack(val,axis=1)


def _numeric_nested(v):
    if v is None or isinstance(v,bool):
        return False
    if isinstance(v,(int,float,np.number)):
        return bool(np.isfinite(v))
    if isinstance(v,(list,tuple)):
        return bool(v) and all(_numeric_nested(x) for x in v)
    if isinstance(v,dict):
        return bool(v) and all(_numeric_nested(x) for x in v.values())
    return False


def metric_coverage(train,validation,contract):
    coverage={}
    for group in contract['metric_groups']:
        if group['group']=='experiment_specific':
            continue
        for split in (('train',) if group.get('center_only') else ('train','validation')):
            section=train if split=='train' else validation
            for name in group['ids']:
                value=section.get('center_geometry',{}).get(name,section.get(name))
                if name=='per_class_recall':
                    value=[row['accuracy'] for row in section['per_class']]
                ok=_numeric_nested(value)
                if ok:
                    status,reason='measured','canonical standard_metrics'
                elif name in ('R_KL','NLL_KL_simple','NLL_KL_normalized_historical'):
                    status,reason='undefined_or_infinite','KL support incompatibility; no epsilon clipping'
                else:
                    status,reason='not_measured','canonical result unavailable; not replaced with zero'
                coverage[f'{split}.{name}']={'status':status,'reason':reason}
    coverage['validation.center_geometry']={'status':'not_applicable',
        'reason':'Only Train class centers; no held-out center refitting'}
    coverage['test']={'status':'not_applicable','reason':'Test forbidden in E-CH4-14'}
    return coverage


def report_graph(graph,z,family,train_y,validation_y,*,data_hashes,subset_id):
    """The scorer is independent of true Validation labels until standard_metrics."""
    tr,va=by_hypothesis(graph,z,family)
    ty=np.asarray(train_y,dtype=np.int64)
    vy=np.asarray(validation_y,dtype=np.int64)
    if (tr.shape!=(len(ty),graph.k,graph.k) or
            va.shape!=(len(vy),graph.k,graph.k) or
            not np.array_equal(ty,graph.labels.detach().cpu().numpy())):
        raise ValueError('Unexpected Train/Validation row/label layout')
    if set(ty.tolist())!=set(range(graph.k)):
        raise ValueError('All K classes required for Train centers')
    # Class-c center comes only from Train representations under candidate c.
    centers=np.stack([tr[ty==c,c,:].mean(axis=0) for c in range(graph.k)])
    train=standard_metrics(tr,centers,ty,.005,split='train')
    validation=standard_metrics(va,centers,vy,.005,split='validation')
    for part in (train,validation):
        part['per_class_recall']=[r['accuracy'] for r in part['per_class']]
    contract_path=ROOT/'code/experiments/COMMON_METRICS_CONTRACT.json'
    contract=json.loads(contract_path.read_text('utf8'))
    if contract['schema']!='chapter4-common-evaluation-contract-v2':
        raise ValueError('Common metrics contract changed; review required')
    coverage=metric_coverage(train,validation,contract)
    return {'schema':'e-ch4-14-metrics-v1','subset_id':subset_id,'family':family,
            'objective':'softplus_JS_tau005','test_accessed':False,
            'center_source':'train_only','metric_contract':contract['schema'],
            'data_hashes':data_hashes,'centers':centers.tolist(),
            'train':train,'validation':validation,'coverage':coverage}
