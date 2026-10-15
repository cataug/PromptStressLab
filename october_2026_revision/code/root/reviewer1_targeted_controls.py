#!/usr/bin/env python3
"""One-model targeted reproducibility controls for Reviewer 1.
Uses existing local weights, datasets, saved outputs; never modifies original runs.
Foreground run, live progress; no package installation or background work.
"""
from __future__ import annotations
import argparse, gc, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'scripts'))
from psl_common import build_prompt, compact_gold, extract_json_object, read_jsonl
from run_model_experiment import load_model, load_tokenizer, render_chat
from evaluate_experiment import entity_tuples, gold_entity_tuples, partial_entity_f1
from compute_psi_volatility import extract_fields, DATASET_FIELDS, field_drift

def get_predictions(model_id):
    out={}
    f=ROOT/'outputs/generations'/model_id/'predictions.jsonl'
    for v in read_jsonl(f):
        if v.get('condition') in [f'P{i}' for i in range(1,7)]:
            out[(v['dataset'],str(v['record_id']),v['condition'])]=v
    return out

def find_records(jobs):
    out={}
    for src in sorted({j['source_file'] for j in jobs}):
        for row in read_jsonl(Path(src)):
            out[(row['dataset'],str(row['record_id']),row['split'])]=row
    return out

def predict(model,tokenizer,model_id,prompt,max_new):
    import torch
    rendered=render_chat(tokenizer,prompt,model_id)
    enc=tokenizer(rendered,return_tensors='pt',add_special_tokens=False)
    device=next(model.parameters()).device
    enc={k:v.to(device) for k,v in enc.items()}
    with torch.inference_mode():
        output=model.generate(**enc,max_new_tokens=int(max_new),do_sample=False,
                        use_cache=True,pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True)
    tokens=output.sequences[0,int(enc['input_ids'].shape[1]):]
    values=tokens.tolist()
    ends=set([tokenizer.eos_token_id])
    if getattr(model,'generation_config',None) is not None:
        eos=model.generation_config.eos_token_id
        if isinstance(eos,(tuple,list,set)):ends.update(eos)
        elif eos is not None:ends.add(eos)
    eos_pos=next((i for i,t in enumerate(values) if t in ends),None)
    real_count=(eos_pos+1) if eos_pos is not None else len(values)
    stop='eos' if eos_pos is not None else ('length' if real_count>=max_new else 'other')
    text=tokenizer.decode(tokens[:real_count],skip_special_tokens=True).strip()
    parsed,_=extract_json_object(text)
    del output,enc,tokens
    return {'parsed_output':parsed,'raw_output':text,'actual_output_tokens':real_count,'stop_reason':stop}

def f1(pred,record):
    return float(partial_entity_f1(entity_tuples(pred),gold_entity_tuples(record))['f1'])

def span_set(pred):
    return set(entity_tuples(pred))

def jac(a,b):
    A=span_set(a);B=span_set(b)
    return 1-len(A&B)/len(A|B) if A|B else 0.

def write_jsonl(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as h:
        for obj in items:
            h.write(json.dumps(obj,ensure_ascii=False)+'\n')

def fallback_alt_demos(ds,eval_ids,exclude,seed=42):
    candidates=[]
    # The normalized train split is the only acceptable source, even if no data can be found.
    for f in (ROOT/'data/normalized').rglob('*.jsonl'):
        # Split may be a row attribute rather than a directory/file name.
        for r in read_jsonl(f):
            if r.get('dataset')!=ds or str(r.get('split','')).lower()!='train':continue
            rid=str(r.get('record_id',''))
            if rid in eval_ids or rid in exclude:continue
            gold=compact_gold(r)
            text=r.get('input_text','')
            if 15<=len(text)<=4000 and 1<=len(gold['entities'])<=40:
                candidates.append((rid,dict(record_id=rid,input_text=text,output=gold)))
    candidates.sort(key=lambda r:r[0])
    if len(candidates)<3:
        raise RuntimeError('Cannot select three independent training demonstrations: train split not found or too small')
    indices=np.random.default_rng(seed).choice(len(candidates),3,replace=False)
    return [candidates[int(i)][1] for i in indices]

def compute_encoder_psi(pairs,model_path,model_kind):
    # pairs: list of (doc, label, a,b) parsed predictions; texts are embedded as short items.
    from transformers import AutoModel, AutoTokenizer
    import torch
    fields=[]; strings=set()
    for doc,ds,a,b in pairs:
        x,_j,_s=extract_fields({'parsed_output':a,'validation':{'json_valid':1,'schema_valid':1}},ds)
        y,_j,_s=extract_fields({'parsed_output':b,'validation':{'json_valid':1,'schema_valid':1}},ds)
        fields.append((x,y,ds))
        for z in (x,y):
            for vals in z.values():strings.update(vals)
    strings=sorted(strings)
    if not strings:return np.zeros(len(pairs))
    tok=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=True)
    if model_kind=='clip':
        from transformers import CLIPModel
        mdl=CLIPModel.from_pretrained(str(model_path),local_files_only=True).eval()
    else:
        mdl=AutoModel.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=True).eval()
    device='cpu'
    mdl.to(device)
    cache={}
    for n in range(0,len(strings),64):
        b=strings[n:n+64]
        x=tok(b,padding=True,truncation=True,max_length=77 if model_kind=='clip' else 256,return_tensors='pt')
        with torch.inference_mode():
            if model_kind=='clip':
                feat=mdl.get_text_features(**x)
                if not isinstance(feat,torch.Tensor):feat=feat.pooler_output if hasattr(feat,'pooler_output') else feat.text_embeds
            else:
                h=mdl(**x).last_hidden_state
                mask=x['attention_mask'].unsqueeze(-1)
                feat=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        feats=torch.nn.functional.normalize(feat.float(),p=2,dim=1).numpy()
        cache.update(zip(b,feats))
        print('[ENCODER]',model_kind,min(n+64,len(strings)),'/',len(strings),flush=True)
    scores=[]
    for x,y,ds in fields:
        scores.append(float(np.mean([field_drift(x[f],y[f],cache) for f in DATASET_FIELDS[ds]])))
    del mdl
    return np.array(scores)

p=argparse.ArgumentParser()
p.add_argument('--model-id',default='qwen3_8b')
p.add_argument('--noise-documents',type=int,default=12)
p.add_argument('--demo-documents',type=int,default=32)
p.add_argument('--clip-path',default='/home/tahiti/Forensics/models/clip_vit_base_patch32')
p.add_argument('--text-encoder-path',default='')
p.add_argument('--skip-generation',action='store_true')
a=p.parse_args()
config=json.loads((ROOT/'config/experiment.json').read_text())
model_cfg=next(v for v in config['models'] if v['model_id']==a.model_id)
manifest=json.loads((ROOT/'manifests/demo_manifest.json').read_text())
jobs=[j for j in read_jsonl(ROOT/'manifests/experiment_jobs.jsonl') if j['model_id']==a.model_id]
records=find_records(jobs)
preds=get_predictions(a.model_id)
out=ROOT/'outputs/statistics/reviewer1_round2/targeted_controls'
out.mkdir(parents=True,exist_ok=True)
selected=[j for j in jobs if j['dataset']=='SciERC' and j['condition']=='P3']
selected.sort(key=lambda j:str(j['record_id']))
selected=selected[:a.noise_documents]
selected_demo=[j for j in jobs if j['dataset']=='EBM-NLP' and j['condition']=='P5']
selected_demo.sort(key=lambda j:str(j['record_id']))
selected_demo=selected_demo[:a.demo_documents]
old_ex={str(e['record_id']) for e in manifest['datasets']['EBM-NLP']}
eval_ids={str(j['record_id']) for j in jobs if j['dataset']=='EBM-NLP'}
alt=fallback_alt_demos('EBM-NLP',eval_ids,old_ex)
(out/'alternative_demonstrations.json').write_text(json.dumps(alt,ensure_ascii=False,indent=2))
print('[ALT DEMOS]',[v['record_id'] for v in alt],flush=True)
noisefile=out/'same_condition_repeats.jsonl'
demofile=out/'alternative_demo_predictions.jsonl'
noise_rows=read_jsonl(noisefile) if noisefile.exists() else []
demo_rows=read_jsonl(demofile) if demofile.exists() else []
noise_done={(z['record_id'],z['rep']) for z in noise_rows}
demo_done={z['record_id'] for z in demo_rows}
if not a.skip_generation:
    print('[LOAD LLM]',a.model_id,flush=True)
    import torch
    tok=load_tokenizer(Path(model_cfg['path']))
    model=load_model(a.model_id,Path(model_cfg['path']))
    for j in selected:
        key=(j['dataset'],str(j['record_id']),j['split'])
        record=records[key]
        prompt=build_prompt(dataset=j['dataset'],record=record,condition=j['condition'],
                  demos=manifest['datasets'][j['dataset']],relation_types=manifest['relation_types'][j['dataset']])
        for rep in (1,2):
            if (str(j['record_id']),rep) in noise_done:continue
            result=predict(model,tok,a.model_id,prompt,j['max_new_tokens'])
            result.update(record_id=str(j['record_id']),dataset=j['dataset'],condition=j['condition'],rep=rep,
                          model_id=a.model_id,f1=f1(result['parsed_output'],record))
            noise_rows.append(result)
            write_jsonl(noisefile,noise_rows)
            print('[NOISE]',len(noise_rows),'/',2*len(selected),'id',j['record_id'],'rep',rep,'stop',result['stop_reason'],flush=True)
    for j in selected_demo:
        if str(j['record_id']) in demo_done:continue
        record=records[(j['dataset'],str(j['record_id']),j['split'])]
        prompt=build_prompt(dataset='EBM-NLP',record=record,condition='P5',demos=alt,
                  relation_types=manifest['relation_types']['EBM-NLP'])
        result=predict(model,tok,a.model_id,prompt,j['max_new_tokens'])
        original=preds.get((j['dataset'],str(j['record_id']),'P5'))
        result.update(record_id=str(j['record_id']),dataset='EBM-NLP',condition='P5_alt',model_id=a.model_id,
                     f1=f1(result['parsed_output'],record),
                     original_P5_f1=f1(original.get('parsed_output') if original else None,record))
        demo_rows.append(result);write_jsonl(demofile,demo_rows)
        print('[ALT DEMO]',len(demo_rows),'/',len(selected_demo),'delta F1',round(result['f1']-result['original_P5_f1'],3),flush=True)
    del model,tok
    gc.collect();torch.cuda.empty_cache()

# Registered after actual generation, not placeholder.
summary={'model_id':a.model_id,'noise_documents':0,'alternative_demo_documents':0,'decoding_note':'Additional single-item runs are not batched like original Phase II; alternative demo comparison may be confounded by inference batch scheduling.'}
noise_by={}
for z in noise_rows:noise_by.setdefault(str(z['record_id']),{})[z['rep']]=z
pairs=[];delta=[];stop=[]
for rid,reps in noise_by.items():
    if 1 not in reps or 2 not in reps:continue
    l,r=reps[1],reps[2]
    pairs.append((rid,'SciERC',l['parsed_output'],r['parsed_output']))
    delta.append(jac(l['parsed_output'],r['parsed_output']))
    stop.extend([l['stop_reason'],r['stop_reason']])
summary['noise_documents']=len(pairs)
summary['same_condition_jaccard_mean']=float(np.mean(delta)) if delta else None
summary['same_condition_jaccard_nonzero_fraction']=float(np.mean(np.array(delta)>0)) if delta else None
summary['same_condition_stop_reasons']={x:stop.count(x) for x in sorted(set(stop))}
if a.clip_path and pairs and Path(a.clip_path).exists():
    print('[CLIP same-condition PSI]',flush=True)
    v=compute_encoder_psi(pairs,Path(a.clip_path),'clip')
    summary['same_condition_clip_psi_mean']=float(np.mean(v))
    summary['same_condition_clip_psi_p95']=float(np.quantile(v,.95))
    pd.DataFrame({'record_id':[p[0] for p in pairs],'same_condition_psi':v}).to_csv(out/'same_condition_psi.csv',index=False)
    # Compare on the identical documents to the original CLIP P3->P4 prompt change.
    baseline_path=ROOT/'outputs/statistics/psi_pairwise.csv'
    if baseline_path.is_file():
        old=pd.read_csv(baseline_path)
        old=old[(old['model_id']==a.model_id)&(old['dataset']=='SciERC')&
                (old['transition']=='P3->P4')].copy()
        look=dict(zip(old['record_id'].astype(str),old['psi']))
        joined=[(float(v[i]),float(look[str(pair[0])])) for i,pair in enumerate(pairs)
                if str(pair[0]) in look]
        if joined:
            arr=np.asarray(joined)
            summary['prompt_change_psi_same_docs_mean']=float(np.mean(arr[:,1]))
            summary['prompt_change_above_noise_fraction']=float(np.mean(arr[:,1]>arr[:,0]))
            summary['prompt_change_psi_to_noise_psi_ratio']=float(np.mean(arr[:,1])/
                np.mean(arr[:,0])) if np.mean(arr[:,0])>1e-9 else None
            summary['same_document_comparisons']=len(joined)

if demo_rows:
    x=np.array([v['f1']-v['original_P5_f1'] for v in demo_rows]);nz=x[x!=0]
    summary['alternative_demo_documents']=len(x)
    summary['alternative_demo_mean_delta_f1']=float(x.mean())
    summary['alternative_demo_wilcoxon_p']=float(wilcoxon(nz).pvalue) if len(nz)>0 else 1.

# Text-only encoder check: all 5 adjacent transitions of Qwen-SciERC; rerun neither LLM nor CLIP.
if a.text_encoder_path:
    assert Path(a.text_encoder_path).exists(), 'Text-only encoder must already exist locally; no downloads or invented measurements'
    print('[TEXT ENCODER CHECK]',a.text_encoder_path,flush=True)
    q=[];xs=[];labels=[]
    for j in selected:
        rid=str(j['record_id'])
        gold=records[(j['dataset'],rid,j['split'])]
        for k in range(1,6):
            l=preds.get(('SciERC',rid,f'P{k}'));r=preds.get(('SciERC',rid,f'P{k+1}'))
            if not l or not r:continue
            q.append((rid,'SciERC',l.get('parsed_output'),r.get('parsed_output')))
            xs.append(abs(f1(l.get('parsed_output'),gold)-f1(r.get('parsed_output'),gold)))
            labels.append(f'{rid} P{k}->P{k+1}')
    psis=compute_encoder_psi(q,Path(a.text_encoder_path),'text')
    pd.DataFrame({'transition':labels,'text_encoder_psi':psis,'abs_delta_f1':xs}).to_csv(out/'text_encoder_check.csv',index=False)
    summary['text_encoder_n_transitions']=len(psis)
    summary['text_encoder_spearman_psi_abs_delta_f1']=float(spearmanr(psis,xs).statistic) if len(psis)>3 else None
(out/'targeted_controls_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print('\n[RESULTS]',json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
print('[WROTE]',out/'targeted_controls_summary.json',flush=True)
