#!/usr/bin/env python3
"""Phase I independent, zero-LLM-call reanalysis of original PromptEng26 data.

Outputs immutable-source audit + corrected TF-IDF metrics + publication figures.
Requires only pandas,numpy,scipy,scikit-learn,matplotlib,openpyxl (existing env).
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

P = ['P1','P2','P3','P4','P5','P6']
FIELDS = ['research_objective','methodology','data_or_material','key_results',
          'application_domain','limitations_or_future_work','keywords']
LABELS={'concat':'Concatenated output','research_objective':'Research objective',
        'methodology':'Methodology','data_or_material':'Data / material',
        'key_results':'Key results','application_domain':'Application domain',
        'limitations_or_future_work':'Limitations / future work','keywords':'Keywords'}
COLORS={'P2':'#4c78a8','P3':'#f58518','P4':'#54a24b','P5':'#b279a2','P6':'#e45756'}


def log(*x):print(*x,flush=True)

def value(v):
    if v is None:return ''
    if isinstance(v,str):return ' '.join(v.split())
    if isinstance(v,list):return '; '.join(' '.join(x.split()) for x in v if isinstance(x,str) and x.strip())
    return str(v)

def concat(obj):
    parts=[f'{f}: {value(obj.get(f))}' for f in FIELDS[:-1] if value(obj.get(f))]
    kw=value(obj.get('keywords'))
    if kw:parts.append('keywords: '+kw)
    return '\n'.join(parts)

def normalize_id(v):
    # Preserve identifiers by using exact source row IDs; avoid coercing to float
    return str(v).strip()

def paired_bootstrap_quantile(matrix, stat, reps=1000, seed=42):
    """Bootstrap independent DOCUMENTS; matrix rows=documents."""
    rng=np.random.default_rng(seed)
    n=len(matrix)
    if n<3:return (np.nan,np.nan)
    vals=[]
    for _ in range(reps):
        take=rng.integers(0,n,n)
        data=matrix[take]
        vals.append(float(stat(data)))
    return tuple(np.quantile(vals,[.025,.975]).tolist())

def holm(p):
    a=np.asarray(p,float); out=np.ones_like(a)
    good=np.flatnonzero(np.isfinite(a)); order=good[np.argsort(a[good])]
    if len(order):
        adjusted=np.maximum.accumulate(a[order]*(len(order)-np.arange(len(order))))
        out[order]=np.minimum(adjusted,1.)
    return out

def wilcoxon_rbc(delta):
    x=np.asarray(delta,float);x=x[np.isfinite(x)&(np.abs(x)>1e-12)]
    if not len(x):return 1.0,0.0
    _,p=stats.wilcoxon(x,zero_method='wilcox',alternative='two-sided')
    ranks=stats.rankdata(abs(x))
    pos=ranks[x>0].sum();neg=ranks[x<0].sum()
    return float(p),float((pos-neg)/(pos+neg))


def discover(legacy, data):
    paths={'run':legacy/'run_summary.xlsx','stable':legacy/'stability_metrics.xlsx',
           'json':legacy/'json_metrics.xlsx','filtered':legacy/'filtered.xlsx'}
    for k, p in paths.items():
        if not p.is_file():raise FileNotFoundError(f'{k}: {p}')
    for lvl in P:
        d=data/lvl
        if not d.is_dir():raise FileNotFoundError('Missing: '+str(d))
    run=pd.read_excel(paths['run'],sheet_name='per_call')
    stable=pd.read_excel(paths['stable'],sheet_name='per_call_metrics')
    jm=pd.read_excel(paths['json'],sheet_name='json_metrics_per_call')
    assert len(run)==3180, f'Unexpected historical records: {len(run)} (expected 3180)'
    assert all(int((run.prompt_level==p).sum())==530 for p in P)
    assert run.abstract_id.nunique()==530
    # resolve old Windows paths by filename; don't trust path rooted on old machine
    records={}; problems=[]
    for row in run.itertuples(index=False):
        aid=normalize_id(row.abstract_id);lv=str(row.prompt_level)
        old=str(getattr(row,'output_parsed_path','')).replace('\\','/')
        name=Path(old).name
        candidate=data/lv/name
        if not candidate.is_file():
            match=list((data/lv).glob(f'row{int(row.row_index)}_*__parsed.json')) if pd.notna(row.row_index) else []
            if len(match)!=1:problems.append((aid,lv,name,len(match)));continue
            candidate=match[0]
        try: obj=json.loads(candidate.read_text(encoding='utf-8-sig'))
        except Exception as e:problems.append((aid,lv,name,str(e)));continue
        if not isinstance(obj,dict):problems.append((aid,lv,name,'not dict'));continue
        records[(aid,lv)]=obj
    if problems:
        log('[ERR] Missing or malformed parsed responses:',problems[:15]);raise RuntimeError(f'Parsed output errors: {len(problems)}')
    assert len(records)==3180
    for aid in set(k[0] for k in records):
        if any((aid,lv) not in records for lv in P):raise RuntimeError('Missing paired conditions for '+aid)
    log('[OK] Loaded all 3180 parsed outputs, 530 complete paired documents')
    return paths,run,stable,jm,records


def calc_tfidf(records, out):
    """One common legacy-compatible TF-IDF vocabulary across 25,440 field/concat texts.

    Reproduces Assess.ipynb ngrams=(1,2), min_df=1. No densification, so memory-safe.
    Corrected field metrics: both blank = 0; one blank = 1; otherwise 1-cosine.
    Conditional metric: NaN if either side blank. All-field PSI is defined for all pairs.
    """
    aids=sorted(set(a for a,p in records),key=lambda s:(len(s),s))
    keys=[(aid,lv,f) for aid in aids for lv in P for f in ['concat']+FIELDS]
    texts=[concat(records[(a,p)]) if f=='concat' else value(records[(a,p)].get(f)) for a,p,f in keys]
    log('[TF-IDF] Fitting sparse unigram+bigram vocabulary on',len(texts),'items...')
    v=TfidfVectorizer(ngram_range=(1,2),min_df=1, dtype=np.float32)
    X=v.fit_transform(texts)
    lut={k:i for i,k in enumerate(keys)}
    log('[TF-IDF] Shape:',X.shape,'nnz:',X.nnz,'vocabulary:',len(v.vocabulary_))
    rows=[]
    for aid in aids:
        for i in range(1,len(P)):
            u,w=P[i-1],P[i]
            rec={'abstract_id':aid,'from':u,'prompt_level':w,'transition':u+'->'+w}
            prev=records[(aid,u)];cur=records[(aid,w)]
            for f in ['concat']+FIELDS:
                k1=(aid,u,f);k2=(aid,w,f)
                a=lut[k1];b=lut[k2]
                p1=bool(texts[a]);p2=bool(texts[b])
                sim=float(X[a].multiply(X[b]).sum()) if (p1 and p2) else 0.
                sim=max(0.,min(1.,sim))
                distance=1-sim if p1 and p2 else (1.0 if p1!=p2 else 0.0)
                rec['psi_'+f]=distance
                rec['psi_'+f+'_conditional']=(1-sim if p1 and p2 else np.nan)
                if f!='concat':
                    rec['disappear_'+f]=int(p1 and not p2)
                    rec['appear_'+f]=int(p2 and not p1)
                    rec['both_empty_'+f]=int(not p1 and not p2)
            rec['any_disappearance']=int(any(rec['disappear_'+f] for f in FIELDS))
            rec['any_appearance']=int(any(rec['appear_'+f] for f in FIELDS))
            rows.append(rec)
        if len(rows)%500==0:log('[TF-IDF] progress:',len(rows),'transitions')
    df=pd.DataFrame(rows)
    assert len(df)==2650
    df.to_csv(out/'phase1_corrected_transitions.csv',index=False)
    vols=df.groupby('abstract_id',sort=True)[['psi_'+f for f in ['concat']+FIELDS]].mean().reset_index()
    vols=vols.rename(columns={'psi_'+f:'vol_'+f for f in ['concat']+FIELDS})
    vols.to_csv(out/'phase1_corrected_volatility.csv',index=False)
    # conditional vol: mean over valid pairs, not fixed five denominator
    cond=df.groupby('abstract_id',sort=True)[['psi_'+f+'_conditional' for f in FIELDS]].mean().reset_index()
    cond.to_csv(out/'phase1_conditional_volatility.csv',index=False)
    return df,vols,cond


def summarize(run, stable, jm, trans, vol, out):
    from_columns=['prompt_level','usage_prompt_tokens','usage_completion_tokens',
                  'usage_total_tokens','latency_seconds','json_valid','json_recovered']
    r=run[from_columns].copy()
    for c in from_columns[1:]:r[c]=pd.to_numeric(r[c],errors='coerce')
    tokens=r.groupby('prompt_level',sort=False).agg(
        n=('json_valid','size'),json_valid_rate=('json_valid','mean'),
        recovery_flag_rate=('json_recovered','mean'),
        mean_prompt_tokens=('usage_prompt_tokens','mean'),
        mean_completion_tokens=('usage_completion_tokens','mean'),
        mean_total_tokens=('usage_total_tokens','mean'),
        median_latency_s=('latency_seconds','median'),mean_latency_s=('latency_seconds','mean')).reset_index()
    # json_recovered has legacy non-trivial semantics; do not equate it with failure
    tokens.to_csv(out/'phase1_tokens_json_latency.csv',index=False)
    arr=[]
    for lvl in P[1:]:
        s=trans.loc[trans.prompt_level==lvl]
        for f in ['concat']+FIELDS:
            val=s['psi_'+f].dropna().to_numpy(float)
            conditional=s['psi_'+f+'_conditional'].dropna().to_numpy(float) if f!='concat' else val
            lo,hi=paired_bootstrap_quantile(val, lambda x:np.mean(x),reps=1200,seed=2026)
            arr.append({'transition':P[P.index(lvl)-1]+'->'+lvl,'prompt_level':lvl,
                        'field':f,'n':len(val),'mean':float(np.mean(val)),
                        'median':float(np.median(val)),'p95':float(np.quantile(val,.95)),
                        'mean_ci_low':lo,'mean_ci_high':hi,'conditional_n':len(conditional),
                        'conditional_mean':float(np.mean(conditional)) if len(conditional) else np.nan})
    pd.DataFrame(arr).to_csv(out/'phase1_psi_summary.csv',index=False)
    vol_stat=[]
    for f in ['concat']+FIELDS:
        val=vol['vol_'+f].to_numpy(float)
        lo,hi=paired_bootstrap_quantile(val,lambda x:np.mean(x),reps=1200,seed=2026)
        vol_stat.append({'field':f,'n_documents':len(val),'mean':val.mean(),
                         'median':np.median(val),'p95':np.quantile(val,.95),
                         'mean_ci_low':lo,'mean_ci_high':hi})
    pd.DataFrame(vol_stat).to_csv(out/'phase1_volatility_summary.csv',index=False)
    ids=trans.groupby('prompt_level').agg(n=('abstract_id','size'),
                 any_disappearance=('any_disappearance','mean'),any_appearance=('any_appearance','mean')).reset_index()
    ids.to_csv(out/'phase1_disappearance_appearance.csv',index=False)
    # family: 4 adjacent comparisons between 5 transition PSI distributions, paired on abstract_id
    pivot=trans.pivot(index='abstract_id',columns='prompt_level',values='psi_concat').reindex(columns=P[1:])
    rows=[]
    for i in range(1,5):
        left,right=P[i],P[i+1]
        delta=(pivot[right]-pivot[left]).to_numpy(float)
        pv,rb=wilcoxon_rbc(delta)
        rows.append({'contrast':right+' minus '+left,'n':len(delta),
                     'mean_delta':float(np.mean(delta)),'wilcoxon_p':pv,'rank_biserial':rb})
    adj=holm([q['wilcoxon_p'] for q in rows])
    for row,a in zip(rows,adj):row['holm_p_within_4_adjacent_transitions']=a
    pd.DataFrame(rows).to_csv(out/'phase1_paired_tests.csv',index=False)
    # Side-by-side check against saved legacy TF-IDF; no rewriting legacy values
    m=stable[stable.prompt_level.isin(P[1:])][['abstract_id','prompt_level','psi_concat_prev']].copy()
    m.abstract_id=m.abstract_id.map(normalize_id)
    chk=m.merge(trans[['abstract_id','prompt_level','psi_concat']],on=['abstract_id','prompt_level'],how='inner',validate='one_to_one')
    chk['difference_recomputed_minus_historic']=chk['psi_concat']-chk['psi_concat_prev']
    chk.to_csv(out/'phase1_legacy_vs_recomputed_concat.csv',index=False)
    legacy_compare=[]
    legacy=stable[stable.prompt_level.isin(P[1:])].copy()
    legacy['abstract_id']=legacy['abstract_id'].map(normalize_id)
    comp=legacy.merge(trans,on=['abstract_id','prompt_level'],how='inner',validate='one_to_one')
    for f in ['concat']+FIELDS:
        col='psi_'+f+'_prev'
        if col not in comp:continue
        a=pd.to_numeric(comp[col],errors='coerce').to_numpy(float)
        b=pd.to_numeric(comp['psi_'+f],errors='coerce').to_numpy(float)
        legacy_compare.append({'field':f,'legacy_mean':float(np.nanmean(a)),
            'corrected_mean':float(np.nanmean(b)),
            'mean_absolute_change':float(np.nanmean(np.abs(a-b))),
            'mean_signed_change':float(np.nanmean(b-a))})
    pd.DataFrame(legacy_compare).to_csv(out/'phase1_null_policy_audit.csv',index=False)
    concat_mae=float(chk['difference_recomputed_minus_historic'].abs().mean())
    log('[AUDIT] Historic-vs-sparse TFIDF concat mean absolute difference:',round(concat_mae,6))
    if concat_mae>0.025:
        log('[WARNING] Nontrivial difference in concatenated PSI: audit legacy code/representation before citing numerical results.')
    log('[SUMMARY] Generated numeric CSV outputs')
    print(tokens.to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
    print(pd.DataFrame(vol_stat)[['field','mean','p95']].to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
    return tokens


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':14,
        'axes.labelsize':13,'xtick.labelsize':11,'ytick.labelsize':11,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white'})

def ecdf(v):
    vals=np.sort(np.asarray(v,float));return vals,np.arange(1,len(vals)+1)/len(vals)

def export(fig,d,name):
    pdf=d/(name+'.pdf');png=d/(name+'.png')
    fig.savefig(pdf,bbox_inches='tight');fig.savefig(png,bbox_inches='tight',dpi=220)
    plt.close(fig);log('[FIG]',pdf)


def figure3(trans,out):
    # Faithful notebook features: 2x3, five ECDF colors, bottom-right inset
    # step histogram with faint fill, common P2..P6 legend.
    fields=['concat','research_objective','methodology','data_or_material','key_results','application_domain']
    fig,axs=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for ax,f in zip(axs.flat,fields):
        ax.set_facecolor('#fbfcfe')
        ia=inset_axes(ax,width='36%',height='35%',loc='lower right',borderpad=0.85)
        ia.set_facecolor('#ffffff')
        for p in P[1:]:
            v=trans.loc[trans.prompt_level==p,'psi_'+f].dropna().to_numpy(float)
            x,y=ecdf(v)
            ax.plot(x,y,color=COLORS[p],lw=2.2,label=p)
            ia.hist(v,bins=np.linspace(0,1,26),density=True,
                    histtype='stepfilled',alpha=.09,color=COLORS[p])
            ia.hist(v,bins=np.linspace(0,1,26),density=True,
                    histtype='step',lw=1.1,color=COLORS[p])
        ax.set(title=LABELS[f],xlabel='PSI (state-aware TF-IDF)',ylabel='ECDF',xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.18,lw=.6)
        ia.set_xlim(0,1);ia.set_xticks([]);ia.set_yticks([])
        for spine in ia.spines.values():spine.set_visible(True);spine.set_linewidth(.65)
    fig.legend(handles=[Line2D([0],[0],color=COLORS[p],lw=2.4,label=p) for p in P[1:]],
        ncol=5,loc='outside lower center',frameon=False)
    export(fig,out,'Figure3_PSI_ECDF_with_insets_corrected')


def figure4(vol,out):
    fields=['concat','research_objective','methodology','data_or_material','key_results','application_domain']
    fig,axs=plt.subplots(2,3,figsize=(18,10),layout='constrained',facecolor='#f2f2f2')
    bins=np.linspace(0,1,21)
    maximum=max(np.histogram(vol['vol_'+f].dropna(),bins=bins)[0].max() for f in fields)
    norm=Normalize(vmin=0,vmax=maximum)
    for ax,f in zip(axs.flat,fields):
        vv=vol['vol_'+f].to_numpy(float)
        counts,_,patches=ax.hist(vv,bins=bins,edgecolor='#202b36',linewidth=.55)
        for height,patch in zip(counts,patches):
            patch.set_facecolor(plt.cm.Blues(.18+.7*norm(height)))
            patch.set_alpha(.94)
        ax.set(title=LABELS[f],xlabel='Mean PSI over five transitions',
               ylabel='Documents',xlim=(0,1))
        ax.set_facecolor('#f7f7f7');ax.grid(axis='y',linestyle='--',alpha=.28,lw=.55)
        ax.axvline(vv.mean(),ls='--',lw=1.5,color='#b44f4b')
        ax.text(.97,.94,f'Mean = {vv.mean():.3f}\nP95 = {np.quantile(vv,.95):.3f}',
                transform=ax.transAxes,ha='right',va='top',fontsize=10,
                bbox=dict(facecolor='white',alpha=.88,edgecolor='#d3dbe2',boxstyle='round,pad=.3'))
    sm=ScalarMappable(norm=norm,cmap=plt.cm.Blues)
    sm.set_array([])
    cb=fig.colorbar(sm,ax=axs.ravel().tolist(),shrink=.68,pad=.012,aspect=30)
    cb.set_label('Count per histogram bin (common scale)')
    export(fig,out,'Figure4_Field_Volatility_gradient_corrected')


def pareto_front(frame):
    # min total tokens and latency; JSON invalidity always zero in recorded logs
    d=frame.sort_values(['usage_total_tokens','latency_seconds'])
    best=float('inf'); keep=[]
    for i,row in d.iterrows():
        y=float(row.latency_seconds)
        if y<best-1e-12:
            keep.append(i);best=y
    return d.loc[keep]


def figure5(run,out):
    """Figure 5: notebook-inspired 2x3, length strata shape legend, full shared axes,
    boxed local-view inset retaining resolution without incomparable axes."""
    r=run.copy()
    for c in ['usage_total_tokens','latency_seconds','estimated_abstract_tokens','json_valid']:
        r[c]=pd.to_numeric(r[c],errors='coerce')
    r=r.dropna(subset=['usage_total_tokens','latency_seconds','estimated_abstract_tokens'])
    q33,q66=r.estimated_abstract_tokens.quantile([1/3,2/3]).tolist()
    r['stratum']=np.select([r.estimated_abstract_tokens<=q33,
        r.estimated_abstract_tokens<=q66],['short','medium'],default='long')
    markers={'short':'o','medium':'^','long':'s'}
    sc={'short':'#4878a8','medium':'#d18b39','long':'#548c70'}
    maxx=float(r.usage_total_tokens.max())*1.03
    maxy=float(r.latency_seconds.max())*1.03
    fig,axs=plt.subplots(2,3,figsize=(19,11),sharex=True,sharey=True,layout='constrained')
    counts=[]
    for ax,lvl in zip(axs.flat,P):
        d=r[r.prompt_level==lvl]
        ax.scatter(d.usage_total_tokens,d.latency_seconds,s=8,alpha=.10,color='#7e8d98',rasterized=True)
        zoom=inset_axes(ax,width='48%',height='46%',loc='upper right',borderpad=.75)
        zoom.set_facecolor('#fcfcfc')
        size={}
        for st in ['short','medium','long']:
            s=d[d.stratum==st]
            front=pareto_front(s)
            size[st]=len(front)
            ax.scatter(front.usage_total_tokens,front.latency_seconds,s=30,
                       marker=markers[st],alpha=.9,edgecolors='#202b36',linewidths=.4,
                       color=sc[st],rasterized=False)
            zoom.scatter(s.usage_total_tokens,s.latency_seconds,s=6,
                         alpha=.12,color=sc[st],rasterized=True)
            zoom.scatter(front.usage_total_tokens,front.latency_seconds,s=24,
                         marker=markers[st],alpha=.95,edgecolors='#202b36',linewidths=.4,color=sc[st])
        # zoom uses per-prompt x-range with quantile-safe padding; never disguise local axis
        x1,x2=np.quantile(d.usage_total_tokens,[.01,.99]);y1,y2=np.quantile(d.latency_seconds,[.01,.99])
        dx=max(1,(x2-x1)*.15);dy=max(.1,(y2-y1)*.15)
        zoom.set_xlim(x1-dx,x2+dx);zoom.set_ylim(max(0,y1-dy),y2+dy)
        zoom.xaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
        zoom.yaxis.set_major_locator(MaxNLocator(nbins=3))
        zoom.tick_params(labelsize=7,pad=1)
        zoom.set_title('Local view',fontsize=9,pad=1)
        ax.set_xlim(0,maxx);ax.set_ylim(0,maxy)
        ax.grid(alpha=.18,lw=.6)
        ax.set_title(f'{lvl} | Pareto: s={size["short"]}  m={size["medium"]}  l={size["long"]}',fontsize=12)
        counts.append({'prompt_level':lvl,'n_short':size['short'],'n_medium':size['medium'],'n_long':size['long']})
    for ax in axs[1,:]:ax.set_xlabel('Total tokens (input + completion)')
    for ax in axs[:,0]:ax.set_ylabel('Latency (seconds)')
    fig.legend(handles=[Line2D([0],[0],marker=markers[s],markersize=8,color='none',
        markerfacecolor=sc[s],markeredgecolor='#202b36',label=label) for s,label in
        [('short',f'Short (≤ Q33={q33:.1f} abstract tokens)'),
         ('medium',f'Medium (Q33–Q66={q66:.1f})'),('long','Long (> Q66)')]],
        ncol=3,loc='outside lower center',frameon=False)
    export(fig,out,'Figure5_Pareto_common_axes_local_insets_corrected')
    pd.DataFrame(counts).to_csv(out/'phase1_pareto_counts.csv',index=False)
    pd.DataFrame([{'Q33_abstract_tokens':q33,'Q66_abstract_tokens':q66,
        'note':'All historical JSON-valid = 1: three-objective Pareto reduces to token/latency 2D'}]).to_csv(out/'phase1_figure5_quantile_definitions.csv',index=False)
    log(f'[STRATA] Q33={q33:.3f}; Q66={q66:.3f}')


def fmt(v,n=3):return '---' if not np.isfinite(v) else f'{v:.{n}f}'

def latex_tables(out):
    p=pd.read_csv(out/'phase1_psi_summary.csv');v=pd.read_csv(out/'phase1_volatility_summary.csv')
    t=pd.read_csv(out/'phase1_tokens_json_latency.csv');q=pd.read_csv(out/'phase1_paired_tests.csv')
    lines=[]
    lines+=['% Phase I: automatically generated from saved original JSON responses',
        '% Legacy embedding=TF-IDF 1-2grams, corrected null-state semantics.',
        '% Source: independent recreation, not OpenAI text-embedding-3-small.',
        '\\begin{table*}[t]','\\centering','\\small',
        '\\caption{Recomputed Phase I concatenated PSI by adjacent prompt transition (state-aware TF--IDF). Confidence intervals are document-bootstrap intervals for the mean; each transition contains 530 paired abstracts.}',
        '\\label{tab:phase1-recovered-psi}','\\begin{tabular}{lrrrrr}','\\toprule',
        'Transition & Mean & Median & P95 & 95\\% CI (mean) & $N$ \\\\', '\\midrule']
    for _,r in p[p.field=='concat'].iterrows():
        lines.append(f'{r.transition} & {r["mean"]:.3f} & {r["median"]:.3f} & {r.p95:.3f} & [{r.mean_ci_low:.3f},{r.mean_ci_high:.3f}] & {int(r.n)} \\\\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table*}','',
        '\\begin{table*}[t]','\\centering','\\small',
        '\\caption{Recomputed Phase I document-level field volatility (mean over five adjacent prompt transitions), with document-bootstrap confidence intervals.}',
        '\\label{tab:phase1-recovered-vol}','\\begin{tabular}{lrrrr}','\\toprule',
        'Field & Mean & Median & P95 & 95\\% CI (mean) \\\\', '\\midrule']
    for _,r in v.iterrows():
        nice=LABELS.get(r.field,r.field).replace('_',' ')
        lines.append(f'{nice} & {r["mean"]:.3f} & {r["median"]:.3f} & {r.p95:.3f} & [{r.mean_ci_low:.3f},{r.mean_ci_high:.3f}] \\\\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table*}','',
        '\\begin{table*}[t]','\\centering','\\small',
        '\\caption{Original recorded Phase I usage and JSON-validity, from \\texttt{run\\_summary.xlsx}. Means are across 530 requests per prompt level.}',
        '\\label{tab:phase1-recovered-tokens}','\\begin{tabular}{lrrrrr}',
        '\\toprule','Prompt & Input tokens & Output tokens & Total tokens & Latency (s) & JSON valid \\\\','\\midrule']
    for _,r in t.iterrows():
        lines.append(f'{r.prompt_level} & {r.mean_prompt_tokens:.1f} & {r.mean_completion_tokens:.1f} & {r.mean_total_tokens:.1f} & {r.mean_latency_s:.2f} & {100*r.json_valid_rate:.1f}\\% \\\\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table*}','',
        '\\begin{table}[t]','\\centering','\\small',
        '\\caption{Paired Wilcoxon tests for adjacent concatenated-PSI transition distributions. Holm adjustment defines one family of four pre-specified tests.}',
        '\\label{tab:phase1-recovered-stats}',
        '\\begin{tabular}{lrrr}','\\toprule','Contrast & $\\Delta$ mean & $p_{\\rm Holm}$ & Rank-biserial \\\\','\\midrule']
    for _,r in q.iterrows():
        lines.append(f'{r.contrast} & {r.mean_delta:+.3f} & {r.holm_p_within_4_adjacent_transitions:.4f} & {r.rank_biserial:+.3f} \\\\')
    lines+=['\\bottomrule','\\end{tabular}','\\end{table}','']
    (out/'phase1_recovered_tables.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'README_PHASE1.txt').write_text('''Original responses: 3180 / 530 abstracts x 6 conditions.
Legacy spreadsheet kept unchanged. Corrected analysis uses global sparse TF-IDF 1-2 grams.
Field PSI state convention: both blank=>0, one blank=>1, both present=>1-cosine.
Conditional PSI excludes both transitions with blank endpoints; compare separate CSV.
Volatility is a per-document mean over five adjacent transitions.
Figure3 preserves ECDF and lower-right histogram insets from Assess.ipynb.
Figure4 preserves blue count-gradient with common count scale from Assess.ipynb.
Figure5 preserves quantile-stratum markers and 2x3 Pareto from Stat.ipynb, but
uses common global axes + visibly marked per-panel zoom insets to address reviewer.
All JSON-valid rates in original logs are 100%; the third Pareto objective is constant.
Do not claim this is OpenAI embedding PSI. Do not infer true prompt-length causality.
No language-model calls, no new training, no pip installs.
''',encoding='utf-8')
    log('[TABLE]',out/'phase1_recovered_tables.tex')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/home/tahiti/PromptStressLab')
    a=ap.parse_args();root=Path(a.root)
    legacy=root/'legacy_PromptEng26'
    if not legacy.exists():legacy=root
    data=root/'phase1_original'/'outputs'
    out=root/'outputs'/'phase1_reanalysis_2026'
    out.mkdir(parents=True,exist_ok=True)
    log('[START] Legacy:',legacy,'/ original JSON:',data,'/ results:',out)
    paths,run,stable,jm,records=discover(legacy,data)
    trans,vol,cond=calc_tfidf(records,out)
    summarize(run,stable,jm,trans,vol,out)
    style();figure3(trans,out);figure4(vol,out);figure5(run,out)
    latex_tables(out)
    log('[OK] Phase I recomputed and figures rendered; no LLM calls.')
if __name__=='__main__':main()
