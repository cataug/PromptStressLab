#!/usr/bin/env python3
"""Explicit opt-in to measure 2026 hosted API repeated same-condition drift.
Call budget default 12 documents x 2 conditions x 3 fresh calls = 72 calls.
No calls occur unless --execute is passed and OPENAI_API_KEY is present.
Never changes old raw outputs. Safe incremental JSONL checkpoints/resume.
"""
import argparse, ast, json, os, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

P=['P3','P6']; FIELDS=['research_objective','methodology','data_or_material','key_results','application_domain','limitations_or_future_work','keywords']

def builder(nb_path,name):
    d=json.loads(Path(nb_path).read_text(encoding='utf8'))
    for cell in reversed(d['cells']):
        if cell.get('cell_type')!='code':continue
        source=''.join(cell['source'])
        if f'def {name}(' not in source or 'SCHEMA_BLOCK' not in source:continue
        try:tree=ast.parse(source)
        except SyntaxError:continue
        nodes=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEMA_BLOCK' for t in node.targets):
                nodes.append(node)
            if isinstance(node,ast.FunctionDef) and node.name==name:
                nodes.append(node)
        if len(nodes)!=2:continue
        ns={'__builtins__':{'range':range,'str':str}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(nb_path),'exec'),ns)
        return ns[name]
    raise RuntimeError('Cannot extract '+name+' from '+str(nb_path))

def flatten(o):
    if not isinstance(o,dict):return ''
    a=[]
    for f in FIELDS:
        v=o.get(f)
        if isinstance(v,list):v='; '.join(str(q).strip() for q in v if isinstance(q,str) and q.strip())
        if isinstance(v,str) and v.strip():a.append(f+': '+' '.join(v.split()))
    return '\n'.join(a)

def parse_response(raw):
    try:
        x=json.loads(raw)
        return x if isinstance(x,dict) else {}
    except Exception:return {}

def ci_doc(df,col,seed=2026,reps=4000):
    groups=df.groupby('abstract_id')[col].mean().to_numpy(float)
    if not len(groups):return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed)
    samples=groups[rng.integers(0,len(groups),(reps,len(groups)))].mean(axis=1)
    return float(groups.mean()),float(np.quantile(samples,.025)),float(np.quantile(samples,.975))

def quantify(rows,out,root):
    # Use EXACTLY the historical 25,440-text TF-IDF vocabulary and IDF weights.
    # This makes prospective distances comparable to the reconstructed Phase I metric,
    # while still NOT claiming to measure historical 2025 serving variability.
    from rebuild_phase1_from_original import discover,concat,value,FIELDS as ALL_FIELDS,P as ALL_PROMPTS
    _,_,_,_,records=discover(root/'legacy_PromptEng26',root/'phase1_original'/'outputs')
    aids=sorted({aid for aid,_ in records},key=lambda s:(len(s),s))
    base=[concat(records[(aid,p)]) if f=='concat' else value(records[(aid,p)].get(f))
          for aid in aids for p in ALL_PROMPTS for f in ['concat']+ALL_FIELDS]
    df=pd.DataFrame(rows)
    assert df.groupby(['abstract_id','condition']).repeat.nunique().min()>=2
    df=df.sort_values(['abstract_id','condition','repeat'])
    texts=[flatten(parse_response(str(x))) for x in df.raw_output]
    vec=TfidfVectorizer(ngram_range=(1,2),dtype=np.float32)
    vec.fit(base)
    x=vec.transform(texts)
    eps=1e-8
    def dist(a,b):
        if not texts[a] and not texts[b]:return 0.0
        if not texts[a] or not texts[b]:return 1.0
        return float(np.clip(1-x[a].multiply(x[b]).sum(),0.,1.))
    idx={(str(r.abstract_id),r.condition,int(r.repeat)):i for i,r in enumerate(df.itertuples(index=False))}
    trials=[]
    aids=sorted(df.abstract_id.astype(str).unique())
    for aid in aids:
        for cond in P:
            repeats=sorted(r for a,c,r in idx if a==aid and c==cond)
            for i,k in enumerate(repeats):
                for j in repeats[i+1:]:
                    trials.append({'abstract_id':aid,'kind':'same_prompt_'+cond,'distance':dist(idx[aid,cond,k],idx[aid,cond,j])})
        pairs=sorted(set(r for a,c,r in idx if a==aid and c=='P3') & set(r for a,c,r in idx if a==aid and c=='P6'))
        for rep in pairs:
            trials.append({'abstract_id':aid,'kind':'P3_vs_P6','distance':dist(idx[aid,'P3',rep],idx[aid,'P6',rep])})
    dt=pd.DataFrame(trials)
    dt.to_csv(out/'same_condition_pairwise_distances.csv',index=False)
    vals=[]
    for kind,g in dt.groupby('kind'):
        mu,lo,hi=ci_doc(g,'distance')
        vals.append({'comparison':kind,'n_documents':g.abstract_id.nunique(),'n_pairs':len(g),'mean_distance':mu,'ci_low':lo,'ci_high':hi,'median_pair_distance':g.distance.median()})
    z=pd.DataFrame(vals).sort_values('comparison')
    z.to_csv(out/'prospective_noise_floor_summary.csv',index=False)
    print('\n[NOISE FLOOR measured on current hosted endpoint]')
    print(z.to_string(index=False,float_format=lambda v:f'{v:.4f}'),flush=True)
    lines=['%% revised - reviewer #1 - comment 8 - Prospective same-prompt repeated-call baseline, NOT retrospective 2025 noise.',
      r'\begin{table}[t]',r'\centering',r'\small',
      r'\caption{Prospective same-condition variation under repeated requests to the current hosted endpoint. Document-bootstrap confidence intervals. These estimates do not reconstruct historical 2025 serving variation.}',
      r'\label{tab:phase1-noise-prospective}',r'\begin{tabular}{lrrr}',r'\toprule',
      r'Comparison & Documents & Mean PSI & 95\% CI \\',r'\midrule']
    for _,q in z.iterrows():
        lines.append(f'{q.comparison.replace("_"," ")} & {int(q.n_documents)} & {q.mean_distance:.3f} & [{q.ci_low:.3f}, {q.ci_high:.3f}] \\\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (out/'prospective_noise_floor_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/home/tahiti/PromptStressLab')
    ap.add_argument('--execute',action='store_true',help='Explicit opt-in to paid hosted API requests')
    ap.add_argument('--documents',type=int,default=12)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--model',default='gpt-4.1-mini')
    a=ap.parse_args();root=Path(a.root)
    if not 2<=a.repeats<=10:raise ValueError('--repeats must be 2..10')
    if not 1<=a.documents<=530:raise ValueError('--documents must be 1..530')
    out=root/'outputs'/'phase1_reanalysis_2026'/'prospective_noise_floor'
    out.mkdir(parents=True,exist_ok=True)
    run=pd.read_excel(root/'legacy_PromptEng26'/'run_summary.xlsx',sheet_name='per_call')
    dataset=pd.read_excel(root/'legacy_PromptEng26'/'filtered.xlsx')
    textcol='Краткое описание' if 'Краткое описание' in dataset.columns else dataset.columns[0]
    uniq=run.drop_duplicates(subset=['abstract_id'])[['abstract_id','row_index']].copy()
    selected=uniq.sample(n=a.documents,random_state=20261015)
    events=[(str(r.abstract_id),int(r.row_index),condition,rep)
            for r in selected.itertuples(index=False) for condition in P for rep in range(a.repeats)]
    random.Random(20261015).shuffle(events)
    budget=len(events)
    print(f'[PLAN] {a.documents} docs x 2 conditions x {a.repeats} repeats = {budget} requests',flush=True)
    print('[MODEL]',a.model,'[OUTPUT]',out,flush=True)
    if not a.execute:
        print('[DRY RUN] No API requests. To explicitly opt in add --execute.',flush=True)
        return
    key=os.environ.get('OPENAI_API_KEY')
    if not key:raise RuntimeError('OPENAI_API_KEY is not set. No calls made.')
    from openai import OpenAI
    nb=root/'legacy_PromptEng26'/'Prompting.ipynb'
    make={c:builder(nb,'build_prompt_'+c.lower()) for c in P}
    rawfile=out/'prospective_calls.jsonl'
    old=[]
    if rawfile.exists():
        for line in rawfile.read_text(encoding='utf8').splitlines():
            try:old.append(json.loads(line))
            except Exception:pass
    seen={(str(d['abstract_id']),d['condition'],int(d['repeat'])) for d in old}
    client=OpenAI(api_key=key)
    max_t=int(pd.to_numeric(run.max_output_tokens,errors='coerce').dropna().mode().iloc[0])
    for idx,(aid,rowid,cond,rep) in enumerate(events,1):
        if (aid,cond,rep) in seen:continue
        abstract=str(dataset.iloc[rowid][textcol]).strip()
        prompt=make[cond](abstract)
        print(f'[API] {idx}/{budget}: doc={aid} condition={cond} repeat={rep}',flush=True)
        for attempt in range(1,4):
            try:
                start=time.perf_counter()
                response=client.responses.create(model=a.model,
                  input=[{'role':'system','content':'You are a precise information extraction assistant.'},
                         {'role':'user','content':prompt}],temperature=0,max_output_tokens=max_t)
                elapsed=time.perf_counter()-start
                record={'abstract_id':aid,'row_index':rowid,'condition':cond,'repeat':rep,
                        'raw_output':response.output_text,'latency_seconds':elapsed,'model':a.model,
                        'api_response_id':getattr(response,'id',None)}
                with rawfile.open('a',encoding='utf8') as f:
                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                old.append(record);seen.add((aid,cond,rep));break
            except Exception as exc:
                print('[RETRY]',attempt,type(exc).__name__,str(exc)[:170],flush=True)
                if attempt==3:raise
                time.sleep(min(2**attempt,8))
    quantify(old,out,root)
    print('[DONE] Completed contemporary repeated-call baseline; no historical outputs touched.',flush=True)
if __name__=='__main__':main()
