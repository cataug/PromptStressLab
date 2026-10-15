#!/usr/bin/env python3
"""Reviewer-1 post-hoc reproducibility analysis; uses existing PromptStressLab predictions only.
Run from repository root: python reviewer1_round2_posthoc.py
No inference, downloads, package installation, or detached execution.
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, binomtest, rankdata, wilcoxon

ROOT=Path.cwd()
MET=ROOT/'outputs/metrics/job_metrics_with_aliases.csv'
PSI=ROOT/'outputs/statistics/psi_pairwise.csv'
OUT=ROOT/'outputs/statistics/reviewer1_round2'
OUT.mkdir(parents=True,exist_ok=True)
for path in (MET,PSI):
    if not path.exists(): raise SystemExit(f'MISSING {path}; run the existing evaluator and PSI script first')
D=pd.read_csv(MET)
S=pd.read_csv(PSI)
MAIN=D[(D['condition'].isin(['P1','P2','P3','P4','P5','P6'])) & (D['physical_or_alias']=='physical')].copy()
ABL=D[D['condition'].isin(['A0','A1','A2','A3','A4','A5','A6'])].copy()
KEY=['model_id','dataset','record_id']
F1='entity_partial_f1'; COLS=['entity_partial_f1','relation_exact_f1','schema_valid','prompt_tokens','output_tokens','seconds_per_item_in_batch','truncated_at_limit','hallucinated_entity_rate']

# Label any empty/empty relation F1 as non-applicable for EBM-NLP.
D.loc[D.dataset=='EBM-NLP','relation_exact_f1']=np.nan
MAIN.loc[MAIN.dataset=='EBM-NLP','relation_exact_f1']=np.nan
ABL.loc[ABL.dataset=='EBM-NLP','relation_exact_f1']=np.nan

# Tables are generated from per-document means, not by pretending alias rows are new predictions.
def aggregate(df):
    z=df.groupby(['model_id','dataset','condition'],as_index=False)[COLS].mean()
    z['n']=df.groupby(['model_id','dataset','condition']).size().to_numpy()
    return z
SUMMARY=aggregate(pd.concat([MAIN,ABL],ignore_index=True))
SUMMARY.to_csv(OUT/'per_condition_full.csv',index=False)

# exact paired binomial McNemar on discordants
def mcnemar(a,b):
    a=np.array(a,dtype=bool); b=np.array(b,dtype=bool)
    b01=int(((~a)&b).sum()); b10=int((a&(~b)).sum())
    p=float(binomtest(min(b01,b10),b01+b10,.5).pvalue) if b01+b10 else 1.
    return b01,b10,p

def paired_wilcoxon(delta):
    x=np.array(delta,dtype=float)
    x=x[np.isfinite(x)]; nz=x[x!=0]
    if len(nz)==0: return (1.,0.,0)
    ranks=rankdata(abs(nz),method='average'); wplus=float(ranks[nz>0].sum()); wminus=float(ranks[nz<0].sum())
    rb=(wplus-wminus)/(wplus+wminus)
    p=float(wilcoxon(nz,zero_method='wilcox',alternative='two-sided').pvalue)
    return p,rb,len(nz)

def holm(p):
    p=np.asarray(p,dtype=float); order=np.argsort(p); out=np.empty(len(p)); maxp=0.
    for i,j in enumerate(order):
        maxp=max(maxp,min(1.,(len(p)-i)*p[j]));out[j]=maxp
    return out

# Six ablation contrasts per model-dataset family; exact same docs matched.
rows=[]
for (m,d),g in ABL.groupby(['model_id','dataset']):
    pivot=g.pivot_table(index='record_id',columns='condition',values=COLS,aggfunc='first')
    if (F1,'A0') not in pivot: continue
    temp=[]
    for k in range(1,7):
        a='A0';b=f'A{k}'
        if (F1,b) not in pivot:continue
        paired=pivot[[ (F1,a),(F1,b),('schema_valid',a),('schema_valid',b),('truncated_at_limit',a),('truncated_at_limit',b)]].dropna()
        x=paired[(F1,b)]-paired[(F1,a)]
        p,rb,nz=paired_wilcoxon(x)
        sb01,sb10,ps=mcnemar(paired[('schema_valid',a)],paired[('schema_valid',b)])
        tb01,tb10,pt=mcnemar(paired[('truncated_at_limit',a)],paired[('truncated_at_limit',b)])
        temp.append(dict(model_id=m,dataset=d,contrast=f'{b}-{a}',n=len(paired),mean_delta_f1=x.mean(),wilcoxon_p=p,rank_biserial=rb,nonzero_diffs=nz,schema_up=sb01,schema_down=sb10,mcnemar_schema_p=ps,limit_up=tb01,limit_down=tb10,mcnemar_limit_p=pt))
    if temp:
        hp=holm([v['wilcoxon_p'] for v in temp]); sh=holm([v['mcnemar_schema_p'] for v in temp]); lh=holm([v['mcnemar_limit_p'] for v in temp]);
        for i,v in enumerate(temp):v.update(holm_wilcoxon_p=hp[i],holm_schema_p=sh[i],holm_limit_p=lh[i]);rows.append(v)
AB_STATS=pd.DataFrame(rows);AB_STATS.to_csv(OUT/'ablation_paired_inference.csv',index=False)

# Direct P6 vs A5: same records, model, dataset.
comb=pd.concat([MAIN[MAIN.condition=='P6'],ABL[ABL.condition=='A5']],ignore_index=True)
rows=[]
for (m,d),g in comb.groupby(['model_id','dataset']):
    pv=g.pivot_table(index='record_id',columns='condition',values=COLS,aggfunc='first')
    if not {(F1,'A5'),(F1,'P6')}.issubset(pv.columns):continue
    q=pv[[(F1,'A5'),(F1,'P6'),('schema_valid','A5'),('schema_valid','P6'),('prompt_tokens','A5'),('prompt_tokens','P6')]].dropna()
    dif=q[(F1,'P6')]-q[(F1,'A5')]; p,rb,nz=paired_wilcoxon(dif)
    sup,sdown,sp=mcnemar(q[('schema_valid','A5')],q[('schema_valid','P6')])
    rows.append(dict(model_id=m,dataset=d,n=len(q),delta_partial_f1=dif.mean(),p=p,rank_biserial=rb,nonzero=nz,delta_input_tokens=(q[('prompt_tokens','P6')]-q[('prompt_tokens','A5')]).mean(),schema_improved=sup,schema_worsened=sdown,mcnemar_schema_p=sp))
P6A5=pd.DataFrame(rows)
if len(P6A5):P6A5['holm_p_across_9_cells']=holm(P6A5.p);P6A5['holm_schema_p_across_9_cells']=holm(P6A5.mcnemar_schema_p)
P6A5.to_csv(OUT/'p6_vs_a5_paired.csv',index=False)

# Phase-II directional IDR: condition-to-condition entity presence, by ontology label.
# Do not confuse field-disappearance with errors; symmetric field appearance is shown.
from collections import Counter
GEN=ROOT/'outputs/generations'
pred={}
for f in GEN.glob('*/predictions.jsonl'):
    with f.open(encoding='utf8') as h:
        for line in h:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('condition') not in ('P1','P2','P3','P4','P5','P6'):continue
            p=r.get('parsed_output') or {}
            ents=defaultdict(set)
            for e in p.get('entities',[]) if isinstance(p.get('entities'),list) else []:
                if isinstance(e,dict) and e.get('text') and e.get('type'):
                    ents[str(e['type']).casefold()].add(' '.join(str(e['text']).split()).casefold())
            rel=set()
            for rr in p.get('relations',[]) if isinstance(p.get('relations'),list) else []:
                if isinstance(rr,dict) and all(rr.get(k) for k in ('head','type','tail')):
                    rel.add(tuple(str(rr[k]).casefold() for k in ('head','type','tail')))
            pred[(r['model_id'],r['dataset'],r['record_id'],r['condition'])]=(dict(ents),rel)
DSETS={'SciERC':['task','method','metric','material','otherscientificterm','generic','relations'], 'EBM-NLP':['participant','intervention','outcome'], 'SciER':['dataset','method','task','relations']}
records=[]
for (m,d,rid),g in MAIN.groupby(KEY):
    v=[];a=[];changed=[];js=[]
    for k in range(1,6):
        l=pred.get((m,d,rid,f'P{k}')); r=pred.get((m,d,rid,f'P{k+1}'))
        if l is None or r is None:continue
        L,LR=l;R,RR=r
        lost=[bool(L.get(f,[]) if f!='relations' else LR) and not bool(R.get(f,[]) if f!='relations' else RR) for f in DSETS[d]]
        found=[not bool(L.get(f,[]) if f!='relations' else LR) and bool(R.get(f,[]) if f!='relations' else RR) for f in DSETS[d]]
        vl=set((lab,x) for lab,values in L.items() for x in values)
        vr=set((lab,x) for lab,values in R.items() for x in values)
        j=1-(len(vl&vr)/len(vl|vr) if vl|vr else 1.)
        changed.append(int(vl!=vr or LR!=RR));v.append(int(any(lost)));a.append(int(any(found)));js.append(float(j))
        records.append(dict(model_id=m,dataset=d,record_id=rid,transition=f'P{k}->P{k+1}',idr=int(any(lost)),appearance=int(any(found)),jaccard_distance=j,changed=int(changed[-1])))
EV=pd.DataFrame(records)
EV.to_csv(OUT/'field_disappearance_jaccard.csv',index=False)
IDR=EV.groupby(['model_id','dataset','transition'],as_index=False)[['idr','appearance','changed','jaccard_distance']].mean()
IDR.to_csv(OUT/'phase2_idr.csv',index=False)

# Document-cluster bootstrap and permutation; identical-output filtering and Jaccard comparator.
T=S.merge(EV,on=['model_id','dataset','record_id','transition'],how='inner',validate='one_to_one')
RNG=np.random.default_rng(20260717)
def robust_rho(x,y):
    if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return np.nan
    return float(spearmanr(x,y).statistic)
def cluster_stats(g,xcol,ycol,changed_only=False,iterations=1000):
    if changed_only:g=g[g.changed==1]
    groups=[v for _,v in g.groupby('record_id')];ng=len(groups)
    if ng<4: return dict(rho=np.nan,low=np.nan,high=np.nan,p=np.nan,n_doc=ng,n_transition=len(g))
    obs=robust_rho(g[xcol].to_numpy(float),g[ycol].to_numpy(float))
    if not np.isfinite(obs):return dict(rho=np.nan,low=np.nan,high=np.nan,p=np.nan,n_doc=ng,n_transition=len(g))
    B=[]; perm=[]
    grouped_x=[p[xcol].to_numpy(float) for p in groups];grouped_y=[p[ycol].to_numpy(float) for p in groups]
    # Cluster permutation uses independent document blocks, preserving within-document transition order.
    # When changed-only filtering makes block lengths differ, permutation uses whole pooled document summary.
    doc_x=np.array([np.mean(z) for z in grouped_x]);doc_y=np.array([np.mean(z) for z in grouped_y]);doc_r=robust_rho(doc_x,doc_y)
    for i in range(iterations):
        idx=RNG.integers(ng,size=ng)
        xx=np.concatenate([grouped_x[j] for j in idx]);yy=np.concatenate([grouped_y[j] for j in idx]); r=robust_rho(xx,yy)
        if np.isfinite(r): B.append(r)
        rr=robust_rho(doc_x,doc_y[RNG.permutation(ng)])
        if np.isfinite(rr):perm.append(rr)
    lo,hi=(np.quantile(B,[.025,.975]) if B else [np.nan,np.nan])
    # Significance for document-level means, intentionally not mislabeled as a transition-pooled test.
    pp=(1+sum(abs(t)>=abs(doc_r) for t in perm))/(len(perm)+1) if perm and np.isfinite(doc_r) else np.nan
    return dict(rho=obs,low=lo,high=hi,p=pp,n_doc=ng,n_transition=len(g),rho_document_means=doc_r)
C=[]
for (m,d),g in T.groupby(['model_id','dataset']):
    for y in ['abs_delta_entity_partial_f1','abs_delta_entity_exact_f1']:
        for xc in ['psi','jaccard_distance']:
            for ch in [False,True]:
                row=cluster_stats(g,xc,y,changed_only=ch)
                C.append(dict(model_id=m,dataset=d,metric=y,measure=xc,changed_only=int(ch),**row))
COR=pd.DataFrame(C)
for (y,xc,ch),indices in COR.groupby(['metric','measure','changed_only']).groups.items():
    pvals=COR.loc[indices,'p'].fillna(1).to_numpy()
    COR.loc[indices,'holm_p']=holm(pvals)
COR.to_csv(OUT/'psi_jaccard_cluster_correlations.csv',index=False)

# Rank agreement across 5 criteria. PSI belongs to transitions, therefore use per-condition
# mean of neighboring transitions (boundary has one neighbor).
psi_lookup={}
for (m,d),g in S.groupby(['model_id','dataset']):
    for k in range(1,7):
        touched=g[(g.left_condition==f'P{k}')|(g.right_condition==f'P{k}')]
        psi_lookup[(m,d,f'P{k}')]=float(touched.psi.mean())
R=[]
for (m,d),g in SUMMARY[SUMMARY.condition.str.startswith('P')].groupby(['model_id','dataset']):
    if len(g)!=6: continue
    g=g.copy();g['neighbor_psi']=[psi_lookup.get((m,d,c),np.nan) for c in g.condition]
    criteria={'f1':('entity_partial_f1',1),'schema':('schema_valid',1),'tokens':('prompt_tokens',-1),'latency':('seconds_per_item_in_batch',-1),'psi':('neighbor_psi',-1)}
    for a,(ca,sgna) in criteria.items():
        for b,(cb,sgnb) in criteria.items():
            if a>=b:continue
            tau,p=kendalltau(sgna*g[ca],sgnb*g[cb])
            R.append(dict(model_id=m,dataset=d,criterion_a=a,criterion_b=b,kendall_tau=tau,p_descriptive=p))
R=pd.DataFrame(R);R.to_csv(OUT/'rq5_rank_agreement.csv',index=False)

# Export LaTeX tables with no new dependencies.
def esc(x):return str(x).replace('_',r'\_')
def tab(title,label,cols,rows,formats=None,long=False):
    name='longtable' if long else 'tabular'
    al='l'*min(3,len(cols))+'r'*max(0,len(cols)-3)
    head=rf'\begin{{{name}}}{{{al}}}'
    lines=[rf'%% revised - reviewer #1 - generated posthoc - {label}',r'\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}',*([] if long else [r'\begin{center}']),head,r'\toprule',' & '.join(cols)+r' \\',r'\midrule']
    for r in rows:lines.append(' & '.join(map(str,r))+r' \\')
    lines += [r'\bottomrule',rf'\end{{{name}}}',rf'\par\smallskip\noindent\textit{{{title}}}',*([] if long else [r'\end{center}']),r'\endgroup', '']
    return '\n'.join(lines)
def fmt(v,prec=3):return '--' if pd.isna(v) else f'{v:.{prec}f}'
TEX=[r'%% Generated only from existing experiment records, not fabricated.',r'\subsection*{Post-hoc condition-level evidence}']
rows=[]
for r in SUMMARY.sort_values(['model_id','dataset','condition']).itertuples():
    if str(r.condition).startswith('A') and r.condition not in ['A5']:continue
    rows.append([esc(r.model_id),esc(r.dataset),esc(r.condition),r.n,fmt(r.prompt_tokens,1),fmt(r.output_tokens,1),fmt(r.entity_partial_f1),fmt(r.relation_exact_f1),fmt(r.schema_valid),fmt(r.truncated_at_limit),fmt(r.hallucinated_entity_rate)])
TEX.append(tab('Mean scores by model, dataset and prompt condition; relation F1 is inapplicable to EBM-NLP.','full_condition', ['Model','Data','Prompt','n','Input','Output','Partial F1','Rel. F1','Schema','Limit','Unsupported'],rows,long=True))
rows=[]
for r in P6A5.sort_values(['model_id','dataset']).itertuples():
    rows.append([esc(r.model_id),esc(r.dataset),r.n,fmt(r.delta_partial_f1),fmt(r.p,4),fmt(r.holm_p_across_9_cells,4),fmt(r.rank_biserial),fmt(r.delta_input_tokens,1),fmt(r.mcnemar_schema_p,4)])
TEX.append(tab('P6--A5 document-paired tests; Holm across the nine dataset--model direct comparisons.','p6a5', ['Model','Data','n','Delta F1','p','Holm p','r_rb','Delta tokens','McN p'],rows))
rows=[]
for r in COR[(COR.metric=='abs_delta_entity_partial_f1')&(COR.changed_only==1)].sort_values(['model_id','dataset','measure']).itertuples():
    rows.append([esc(r.model_id),esc(r.dataset),r.measure,r.n_doc,r.n_transition,fmt(r.rho),fmt(r.low),fmt(r.high),fmt(r.holm_p,4)])
TEX.append(tab('Changed-only partial-F1 associations: document-cluster bootstrap intervals; Holm-adjusted document-mean permutation p-values.','changed_rho', ['Model','Data','Measure','docs','pairs','rho','CI lo','CI hi','Holm p'],rows))
rows=[]
for r in IDR.groupby(['model_id','dataset'],as_index=False)[['idr','appearance','changed']].mean().itertuples():
    rows.append([esc(r.model_id),esc(r.dataset),fmt(r.idr),fmt(r.appearance),fmt(r.changed)])
TEX.append(tab('Mean directional field transitions over five adjacent prompt steps.','phase2idr',['Model','Dataset','Disappear','Appear','Changed'],rows))
rows=[]
for r in R[(R.criterion_a=='f1') | (R.criterion_b=='f1')].sort_values(['model_id','dataset','criterion_a','criterion_b']).itertuples():
    rows.append([esc(r.model_id),esc(r.dataset),r.criterion_a,r.criterion_b,fmt(r.kendall_tau)])
TEX.append(tab('RQ5: Kendall agreement across criterion ranks; negative values indicate trade-offs.','rq5rank',['Model','Data','Criterion A','Criterion B','Kendall tau'],rows,long=True))
# Append every selected demonstration unabridged: the P4/P5/P6 exemplars are
# identical to the manifest, with no editorial rewriting.
manifest_path=ROOT/'manifests/demo_manifest.json'
if manifest_path.exists():
    demo=json.loads(manifest_path.read_text(encoding='utf8'))
    TEX.append(r'\subsection*{All nine fixed Phase II demonstrations (literal manifest)}')
    for dataset, exs in demo.get('datasets',{}).items():
        TEX.append(r'\noindent\textbf{'+esc(dataset)+r'}\par')
        for i,x in enumerate(exs,1):
            TEX.append(r'\noindent\textit{Demonstration '+str(i)+r', ID '+esc(x['record_id'])+r'}')
            block=f"Demonstration {i}\nINPUT:\n{x['input_text']}\nOUTPUT:\n"+json.dumps(x['output'],ensure_ascii=False,separators=(',',':'))
            TEX.append(r'\begin{lstlisting}[style=promptstyle,basicstyle=\ttfamily\scriptsize]')
            TEX.append(block)
            TEX.append(r'\end{lstlisting}')
else:
    TEX.append(r'\noindent\emph{Demonstration manifest was unavailable in this local copy; full literal examples are not verified.}')
tex='\n'.join(TEX)
(ROOT/'reviewer1_posthoc_tables.tex').write_text(tex,encoding='utf8')
print('[OK] reviewed records:',len(D),'transitions:',len(S),'matched:',len(T))
for filename in ['per_condition_full.csv','ablation_paired_inference.csv','p6_vs_a5_paired.csv','phase2_idr.csv','psi_jaccard_cluster_correlations.csv','rq5_rank_agreement.csv']:
    print('[WROTE]',OUT/filename)
print('[WROTE]',ROOT/'reviewer1_posthoc_tables.tex')
