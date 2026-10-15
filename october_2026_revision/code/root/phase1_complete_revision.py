#!/usr/bin/env python3
"""Phase I manuscript patcher + faithful P6 notebook reconstruction + duplicate-call audit.
No network, no packages installed, no overwriting original documents or figures.
Run after rebuild_phase1_from_original.py (distributed alongside this script).
"""
import argparse
import ast
import json
import re
import shutil
from collections import Counter
from pathlib import Path
import pandas as pd

PROMPTS=['P1','P2','P3','P4','P5','P6']
FIELDS=['concat','research_objective','methodology','data_or_material','key_results','application_domain','limitations_or_future_work','keywords']
NAME={'concat':'Concatenated output','research_objective':'Research objective','methodology':'Methodology','data_or_material':'Data/material','key_results':'Key results','application_domain':'Application domain','limitations_or_future_work':'Limitations/future work','keywords':'Keywords'}

def announce(*args):print(*args,flush=True)

def load_notebook_builder(nbfile):
    if not nbfile.exists():raise FileNotFoundError('Original notebook is required: '+str(nbfile))
    d=json.loads(nbfile.read_text(encoding='utf-8'))
    for cell in reversed(d.get('cells', [])):
        if cell.get('cell_type')!='code':continue
        s=''.join(cell.get('source',[]))
        if 'def build_prompt_p6(' not in s or 'SCHEMA_BLOCK' not in s:continue
        try:tree=ast.parse(s)
        except SyntaxError:continue
        schema=None; fn=None
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCHEMA_BLOCK' for t in node.targets):
                schema=ast.literal_eval(node.value)
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='build_prompt_p6':fn=node
        if schema is None or fn is None:continue
        # Execute just the isolated source of one function; not the notebook/cells.
        namespace={'SCHEMA_BLOCK':schema,'__builtins__':{'range':range,'str':str}}
        isolated=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
        exec(compile(isolated,str(nbfile),'exec'),namespace)
        p6=namespace['build_prompt_p6']('<ABSTRACT>')
        assert p6.count('Rule set repetition #')==30
        assert 'Rule set repetition #30' in p6
        return p6
    raise RuntimeError('Could not isolate the complete build_prompt_p6() from Prompting.ipynb')

def discover_tex(root,name):
    p=root/name
    if not p.exists():raise FileNotFoundError('Expected manuscript: '+str(p))
    return p

def replace_once(s,pattern,replacement,desc,flags=re.S):
    new,n=re.subn(pattern,lambda m:replacement,s,count=1,flags=flags)
    if n!=1:raise RuntimeError(f'Cannot reliably identify {desc}; unchanged original retained.')
    return new

def table_rows(src,cond=None):
    if cond is not None:return src.loc[cond(src)]
    return src

def tex_value(x,n=3):return f'{float(x):.{n}f}'

def make_tables(out):
    psi=pd.read_csv(out/'phase1_psi_summary.csv')
    vol=pd.read_csv(out/'phase1_volatility_summary.csv')
    tok=pd.read_csv(out/'phase1_tokens_json_latency.csv')
    ids=pd.read_csv(out/'phase1_disappearance_appearance.csv')
    tst=pd.read_csv(out/'phase1_paired_tests.csv')
    conc=psi[psi.field=='concat'].sort_values('prompt_level')
    parts=[]
    parts.extend([r'%% revised - reviewer #1 - comment 1 - Matched 530-document PSI percentiles, correctly recomputed from preserved JSON.',
        r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Phase I concatenated TF--IDF PSI for adjacent prompt conditions (530 paired abstracts per row). Corrected null-state convention; confidence intervals resample documents.}',
        r'\label{tab:pilot-domain}',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Transition & Mean & Median & P95 & 95\% CI & $N$ \\',r'\midrule'])
    for _,x in conc.iterrows():
        parts.append(f'{x.transition} & {tex_value(x["mean"])} & {tex_value(x["median"])} & {tex_value(x.p95)} & [{tex_value(x.mean_ci_low)}, {tex_value(x.mean_ci_high)}] & {int(x.n)} \\\\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    parts.extend([r'%% revised - reviewer #1 - comment 1 - Field volatility now directly corresponds to Figure 4.',
        r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Phase I document-level volatility computed from five successive state-aware TF--IDF distances.}',
        r'\label{tab:phase1-revised-vol}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Field & Mean & Median & P95 & 95\% CI \\',r'\midrule'])
    for _,x in vol.iterrows():
        parts.append(f'{NAME[x.field]} & {tex_value(x["mean"])} & {tex_value(x["median"])} & {tex_value(x.p95)} & [{tex_value(x.mean_ci_low)}, {tex_value(x.mean_ci_high)}] \\\\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    parts.extend([r'%% revised - reviewer #1 - comment 11 - Recorded API token usage and numerical JSON-validity rates.',
        r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Original Phase I API usage and JSON validity (means across 530 calls per condition). Latency is descriptive.}',
        r'\label{tab:phase1-usage}',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Prompt & Input & Output & Total & Latency (s) & JSON valid \\',r'\midrule'])
    for _,x in tok.iterrows():
        parts.append(f'{x.prompt_level} & {x.mean_prompt_tokens:.1f} & {x.mean_completion_tokens:.1f} & {x.mean_total_tokens:.1f} & {x.mean_latency_s:.2f} & {100*x.json_valid_rate:.1f}\\% \\\\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    parts.extend([r'%% revised - reviewer #1 - comments 7 and 11 - Explicit paired tests and emergence/disappearance semantics.',
        r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Phase I directional field transitions. IDR denotes any-field disappearance and is not an instruction-failure rate.}',
        r'\label{tab:phase1-idr}',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Transition & $N$ & Disappearance & Appearance \\',r'\midrule'])
    for _,x in ids.sort_values('prompt_level').iterrows():
        parts.append(f'{PROMPTS[PROMPTS.index(x.prompt_level)-1]}--{x.prompt_level} & {int(x.n)} & {x.any_disappearance:.3f} & {x.any_appearance:.3f} \\\\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    parts.extend([r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Paired Wilcoxon comparisons of adjacent transition-distance distributions with Holm correction across four prespecified comparisons. Rank-biserial effect sizes are reported without multiplicity correction.}',
        r'\label{tab:phase1-paired}',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Contrast & $\Delta$ mean & $p_{\mathrm{Holm}}$ & $r_{\mathrm{rb}}$ \\',r'\midrule'])
    for _,x in tst.iterrows():
        parts.append(f'{x.contrast.replace(" minus "," $-$ ")} & {x.mean_delta:+.3f} & {x.holm_p_within_4_adjacent_transitions:.4f} & {x.rank_biserial:+.3f} \\\\')
    parts.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (out/'phase1_ALL_INLINE_TABLES.tex').write_text('\n'.join(parts)+'\n',encoding='utf-8')
    return '\n'.join(parts)

def search_repeats(root,run,outputdir):
    # A raw JSON file and its parsed derivative are ONE inference, not two.
    rawroot=root/'phase1_original'/'outputs'
    repeat_rows=[]
    run=run.copy()
    for c in ['abstract_id','prompt_level','run_timestamp']:
        if c not in run:run[c]=''
    for (doc,lv),g in run.groupby(['abstract_id','prompt_level']):
        if len(g)>1:repeat_rows.append({'abstract_id':doc,'condition':lv,'distinct_rows':len(g),'distinct_timestamps':g.run_timestamp.nunique()})
    pd.DataFrame(repeat_rows,columns=['abstract_id','condition','distinct_rows','distinct_timestamps']).to_csv(outputdir/'phase1_same_condition_repeats.csv',index=False)
    files=list(rawroot.glob('P?/*.json'))
    src=[x for x in files if not x.name.endswith('__parsed.json')]
    bykey=Counter((p.parent.name,p.name.split('_')[0]) for p in src)
    multiple=sum(n>1 for n in bykey.values())
    report={'api_call_records':int(len(run)),'unique_doc_condition_pairs':int(run.groupby(['abstract_id','prompt_level']).ngroups),
        'independently_recorded_repeats':len(repeat_rows),'raw_responses':len(src),
        'parsed_derivatives':len(files)-len(src),'filename_duplicate_groups':multiple,
        'noise_floor_identifiable':bool(repeat_rows),
        'note':'Two files (raw and __parsed) are one API generation. Noise floor needs repeated identical requests. Distinct document or different prompt are not replicates.'}
    (outputdir/'phase1_noise_floor_identifiability.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    announce('[NOISE FLOOR]',json.dumps(report,ensure_ascii=False))
    return report

def patch_tex(root,manuscript,out,p6):
    src=discover_tex(root,manuscript)
    s=src.read_text(encoding='utf-8')
    tables=make_tables(out)
    # Only patch a known manuscript structure; never modify user's source file.
    expected=['\\label{fig:pilot-ecdf}','\\label{fig:pilot-volatility}','\\label{fig:pilot-pareto}', '\\label{tab:pilot-domain}', '\\label{app:phase1-prompts-p6}']
    for x in expected:
        if x not in s:raise RuntimeError('This manuscript differs from expected skeleton: missing '+x)
    # Methods phase I: correct the actual embedding backend while preserving citation.
    s=replace_once(s,
        r'For semantic analysis, only successfully parsed JSON outputs.*?For document \$a\$, field \$f\$, and adjacent prompt',
        r'''%% revised - reviewer #1 - comments 1,7 - Correct the recorded backend and null-state rules.
The preserved Phase~I analysis used the TF--IDF fallback (unigrams and
bigrams) from the archived \texttt{Assess.ipynb}; it did not use
\texttt{text-embedding-3-small}~\cite{openaiEmbeddingModels} in this
executed run. Reanalysis fits a single vocabulary to all preserved field
and concatenated texts. When both adjacent representations are empty,
state-aware PSI is zero; when exactly one is empty, it is one; otherwise
it is one minus cosine similarity. Conditional field PSI, defined only
when both fields are present, is also exported. Consequently, Phase~I
PSI is a lexical-output distance rather than a validated contextual
semantic embedding. The source outputs were strictly parsed in the
original log; an explicit null is not embedded as the string `null'.
For document $a$, field $f$, and adjacent prompt''','embedding methodology')
    s=replace_once(s,r'Embedding sensitivity was checked in the original experiment.*?numerically\s+pooled\.',
        r'''%% revised - reviewer #1 - comment 1 - Remove unsupported cross-encoder replication claim.
No verified \texttt{text-embedding-3-large} sensitivity check is
available for the archived Phase~I run. Phase~I TF--IDF values and
Phase~II CLIP values are not numerically pooled.''','embedding claim')
    # Clarify P6 in main section.
    s=replace_once(s,r'An important provenance limitation concerns \$P6\$:.*?of token length\.',
        r'''%% revised - reviewer #1 - comment 2 - Recovered exact executed P6 constructor.
The original \texttt{Prompting.ipynb} contains the complete Phase~I
$P6$ constructor, which repeats seven grounding/formatting rules 30
times before appending five negative examples and a final checklist.
The complete instantiated template, with the target abstract replaced
by a placeholder, appears in Appendix~\ref{app:phase1-prompts-p6}.
Unlike Phase~II $P6$, historical Phase~I $P6$ does not contain the three
$P5$ demonstrations: the phases use different interventions and cannot
be pooled or interpreted as an isolated length experiment.''','main P6 paragraph')
    # Replace the Phase I results subsection as a unit to ensure stale claims disappear.
    first=s.index(r'\subsection{Phase I: controlled stability and cross-domain behavior}')
    last=s.index(r'\subsection{Phase II: extraction correctness and model dependence}',first)
    p=pd.read_csv(out/'phase1_psi_summary.csv');v=pd.read_csv(out/'phase1_volatility_summary.csv')
    tok=pd.read_csv(out/'phase1_tokens_json_latency.csv')
    conc=p[p.field=='concat'].set_index('prompt_level')
    obj=v.set_index('field')
    result=r'''\subsection{Phase I: controlled stability and cross-domain behavior}
%% revised - reviewer #1 - comments 1,2,7,10,11 - Complete same-source recomputation.
All 3,180 saved Phase~I predictions (530 abstracts under six prompt
conditions) were recovered and evaluated with one state-aware TF--IDF
implementation. Both the numerical summaries and Figures~\ref{fig:pilot-ecdf}
--\ref{fig:pilot-pareto} are now generated from the identical records.
The historical source calculation incorrectly assigns maximal distance to
some unchanged empty-field pairs; corrected distances and conditional
(nonempty-to-nonempty) estimates are retained separately.

%% revised - reviewer #1 - comment 1 - Matched summary from JSON.
Concatenated PSI rises from a mean of %.3f for $P1\!\rightarrow\!P2$
to %.3f for $P5\!\rightarrow\!P6$; the corresponding 95th
percentiles are %.3f and %.3f. Document-level volatility is
%.3f for research objective, %.3f for methodology, and %.3f for
key results. These are descriptive TF--IDF distances, not field-level
accuracy scores.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/Figure3_PSI_ECDF_with_insets_corrected.pdf}
\caption{State-aware Phase~I TF--IDF PSI distributions for the 530 paired abstracts.
The six panels share the same horizontal scale; histogram insets show density.}
\label{fig:pilot-ecdf}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/Figure4_Field_Volatility_gradient_corrected.pdf}
\caption{Corrected Phase~I document-level volatility over five prompt transitions.
Histogram bins and the blue count scale are shared across panels; mean and P95
are displayed.}
\label{fig:pilot-volatility}
\end{figure*}

%% revised - reviewer #1 - comment 11 - Q33/Q66 definitions and valid rate.
The preserved \texttt{run\_summary.xlsx} records strict JSON-validity
rates of %.1f\%% for every Phase~I condition. Thus, the original
three-objective Pareto formulation has a constant validity axis and
reduces to a two-objective descriptive token--latency comparison.
Figure~\ref{fig:pilot-pareto} retains the six-panel arrangement and
quantile-stratified markers; the global axes are shared, with explicitly
labelled local insets. Here $s$, $m$, and $l$ denote the short, medium,
and long source-abstract length terciles, separated by Q33 and Q66,
respectively. Cost differences are descriptive because hosted-API timing
was measured only once per document--condition combination.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figs/Figure5_Pareto_common_axes_local_insets_corrected.pdf}
\caption{Phase~I observed total tokens versus latency, with shared main axes
and condition-specific magnified insets. Markers denote source-length
terciles: circle ($s$), triangle ($m$), and square ($l$); Q33 and Q66
are the 33rd and 66th percentiles of abstract tokens. With all outputs
valid JSON, this is a two-objective Pareto view.}
\label{fig:pilot-pareto}
\end{figure*}

%% revised - reviewer #1 - comments 1,7,11 - Fully inline numerical evidence.
%s

The separately described 420-document ACL-domain experiment cannot
be numerically reconstructed from the 530-document \emph{Information
Sciences} archive alone. No values from these distinct corpora are
silently combined, and the original ACL cross-domain estimates remain
unverified unless their source-level outputs are recovered.

'''%(float(conc.loc['P2','mean']),float(conc.loc['P6','mean']),float(conc.loc['P2','p95']),float(conc.loc['P6','p95']),
       float(obj.loc['research_objective','mean']),float(obj.loc['methodology','mean']),float(obj.loc['key_results','mean']),
       float(100*tok.json_valid_rate.min()),tables)
    # Optional independent contemporary same-prompt experiment, only if real outputs exist.
    noise_table=out/'prospective_noise_floor'/'prospective_noise_floor_table.tex'
    if noise_table.is_file():
        result=result.replace('The separately described 420-document ACL-domain experiment',
            '%% revised - reviewer #1 - comment 8 - Actually measured same-condition baseline on current endpoint.\n'
            'A separate prospective repeated-call experiment quantifies same-prompt\n'
            'variation on the current hosted endpoint. It does not retrospectively\n'
            'identify the 2025 deployment noise floor.\n\n'
            +noise_table.read_text(encoding='utf-8')+'\n'
            'The separately described 420-document ACL-domain experiment')
    s=s[:first]+result+s[last:]
    # Replace partial P6 appendix block with the exact 30-repeat template.
    start=s.index(r'\subsection{Overloaded instruction block: P6}')
    b=s.index(r'\begin{promptbox}',start)
    e=s.index(r'\end{promptbox}',b)+len(r'\end{promptbox}')
    newblock=(r'''\subsection{Historical overloaded condition: P6}
\label{app:phase1-prompts-p6}
%% revised - reviewer #1 - comment 2 - Exact code-reconstructed P6, 30 repeated rule blocks.
The following is the exact P6 text produced by the archived
\texttt{build\_prompt\_p6()} function. The runtime abstract has been
replaced by the marker \texttt{<ABSTRACT>}; all 30 numbered repetitions
and negative examples are preserved. It is structurally distinct from
Phase~II P6, which extends Phase~II P5.
\begin{promptbox}{Phase I P6: complete template reconstructed from executed code}
\begin{lstlisting}[style=promptstyle]
'''+p6+'\n'+r'''\end{lstlisting}
\end{promptbox}''')
    s=s[:start]+newblock+s[e:]
    # Replace the outdated historical-constraints appendix paragraph when present.
    historical_start=s.find('A traceable correction of the \x5cemph{historical} Phase~I experiment')
    historical_end=s.find('Same-prompt local repeatability',historical_start)
    if historical_start>=0 and historical_end>historical_start:
        s=s[:historical_start]+r'''%% revised - reviewer #1 - comments 1,2,10,11 - Archive restored and Figures 1,3--5 corrected.
The original Phase~I 530-document archive and the executed P6 constructor
have now been recovered. Recomputed PSI and volatility are reported from the
identical source records in the main text, and Figures~3--5 have been replaced
by calibrated versions. All Phase~I JSON-validity rates are numerical.
The distinct 420-document ACL-domain archive was not part of this recovery.
No original same-document, same-condition repeated API calls were found;
therefore historic run-to-run noise is not identifiable from these
3,180 one-call observations. Figure~1 has been corrected and denotes
IDR as field disappearance, rather than instruction failure.

'''+s[historical_end:]
    # Remove obsolete assertion that earlier figure was still erroneous.
    s=s.replace('Figure~1 still labels IDR as ``instruction failure''.',
                'Figure~1 has been replaced and correctly defines IDR as field disappearance.')
    dest=out/'PromptStressLab_PHASE1_FINAL_INLINE.tex'
    dest.write_text(s,encoding='utf-8')
    # Figure outputs are copied to the original fig directory only when target
    # doesn't exist; manuscript points at explicit distinct new PDF names.
    dstfig=root/'figs';dstfig.mkdir(exist_ok=True)
    for basename in ['Figure3_PSI_ECDF_with_insets_corrected.pdf',
                     'Figure4_Field_Volatility_gradient_corrected.pdf',
                     'Figure5_Pareto_common_axes_local_insets_corrected.pdf']:
        shutil.copy2(out/basename,dstfig/basename)
    announce('[FINAL TEX]',dest, 'bytes=',dest.stat().st_size)
    return dest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/home/tahiti/PromptStressLab')
    ap.add_argument('--tex',default='PromptStressLab_FINAL_R1_all_tables.tex')
    args=ap.parse_args();root=Path(args.root)
    out=root/'outputs'/'phase1_reanalysis_2026'
    for name in ['phase1_psi_summary.csv','phase1_volatility_summary.csv','Figure3_PSI_ECDF_with_insets_corrected.pdf']:
        if not (out/name).exists():raise FileNotFoundError(f'Run original reanalysis first: {out/name}')
    nb=root/'legacy_PromptEng26'/'Prompting.ipynb'
    p6=load_notebook_builder(nb)
    (out/'phase1_P6_FULL_EXACT_TEMPLATE.txt').write_text(p6,encoding='utf-8')
    announce('[P6] complete constructor:',p6.count('Rule set repetition #'),'rule blocks;',len(p6),'characters')
    run=pd.read_excel(root/'legacy_PromptEng26'/'run_summary.xlsx',sheet_name='per_call')
    search_repeats(root,run,out)
    patch_tex(root,args.tex,out,p6)
    announce('[OK] Phase I: numerical revision built; outputs unchanged; no API calls.')
if __name__=='__main__':main()
