#!/usr/bin/env python3
"""Complete the second-round PromptStressLab manuscript from archived *real* results.
Usage: python build_promptstresslab_final.py
Runs offline in the EXISTING environment, does not rerun LLMs, does not edit input.
Outputs one self-contained TeX manuscript (posthoc tables and demonstrations inlined).
"""
from pathlib import Path
from collections import defaultdict
import json, math, re, sys
import numpy as np
import pandas as pd

ROOT=Path.cwd()
BASE=ROOT/'PromptStressLab_R1_round2_reviewer1_seed.tex'
if not BASE.exists(): BASE=ROOT/'PromptStressLab_FINAL_R1_all_tables.tex'
assert BASE.exists(), f'Missing source TeX {BASE}'
s=BASE.read_text(encoding='utf8')
assert r'\section{Record-level post-hoc results}' in s
assert r'\bibliography{sample}' in s

# Critical corrected qualification: prior batch-output width is not item-specific completion length.
old='''Input tokens include the formatted model-specific chat template; output
 tokens are generated token IDs, including any end-of-sequence/padding IDs kept
 in the generated tensor.'''
# Variant in supplied TeX has no leading spaces.
old=old.replace('output\n tokens','output\ntokens').replace('kept\n in','kept\nin')
new=r'''%% revised - reviewer #1 - comments 3,4,13 - Distinguish per-sequence output from batched padded generation width.
Input tokens include the formatted model-specific chat template. The original
\texttt{output\_tokens} counter records the \emph{shared tensor width} for
all items in a batch, including EOS/padding positions after short completions;
therefore it is \emph{not} an unbiased estimate of per-response output length.
For the supplemental cost table, the saved raw responses are separately
re-tokenized with the corresponding frozen model tokenizer, yielding a
\emph{retokenized visible-output length} (an estimate, not exact generated IDs).'''
if old not in s:
    print('[WARN] original token explanation not matched; searched a fallback')
    s=re.sub(r'Input tokens include the formatted model-specific chat template; output\s+tokens are generated token IDs, including any end-of-sequence/padding IDs kept\s+in the generated tensor\.',lambda _:new,s,count=1)
else:s=s.replace(old,new,1)
s=s.replace('''The recorded \\texttt{truncated\\_at\\_limit} flag is
\\(\\mathbb{I}[n_{\\mathrm{generated}}\\geq n_{\\max}]\\), a conservative
\\emph{limit-hit indicator}, not proof that a complete JSON answer was cut off.''','''%% revised - reviewer #1 - comments 4,13 - Correct a batch-level bookkeeping artifact.
The recorded \\texttt{truncated\\_at\\_limit} flag is based on that
padded tensor width, so it is a batch-level cap indicator that can be 1 even
for responses that stopped earlier. It cannot be interpreted as a per-document
truncation rate; confirmed termination reasons were not retained for the
original 8,526 generations.''')
a='''The recorded output-limit-hit rate is separately summarized by model, dataset,
and prompt. It is not synonymous with missing or malformed JSON, because the
extraction parser may recover an earlier complete JSON object and because an
entire padded generation tensor can reach the length cap. Manual EOS inspection
or generation with a larger cap is needed to establish actual truncation.'''
b=r'''%% revised - reviewer #1 - comments 4,13 - Separate padding artifacts from true truncation.
The original cap flag is a batch-width artifact rather than a per-item
truncation measurement. A supplemental observable \emph{incomplete-JSON-suffix}
indicator can be computed from the raw completion, but it too is a proxy,
not a replacement for exact EOS-versus-length-stop flags. Claims about true
truncation remain limited until a per-sequence stopping-reason audit is run.'''
assert a in s;s=s.replace(a,b,1)
start=s.index('%% revised - reviewer #1 - comments 4,13 - Parser recovery vs strict JSON, and saturation of the generation budget.')
end=s.index('\\begin{table}[t]',start)
s=s[:start]+r'''%% revised - reviewer #1 - comments 4,13 - Correct erroneous 100% truncation interpretation after source-code audit.
All 8,526 planned physical generations produced saved records, with 2,842 per
model. A JSON object could be recovered from each saved response, although
this does not establish strict raw-output JSON compliance. The original
\\texttt{output\\_tokens} and \\texttt{truncated\\_at\\_limit} fields came
from each \emph{batched generation tensor}, not from the per-sample endpoint.
Consequently, apparent 100\\% limit-hit values for some Gemma conditions and
the published per-condition output-token means do \emph{not} establish
100\\% actual truncation. We report these original log values only as
legacy diagnostics and provide a separate retokenization-based length audit
in Appendix~\\ref{app:expanded-r1} when tokenizer files are accessible.
Any low F1 may reflect extraction errors, truncation, or both; a definitive
attribution requires per-response stopping reasons.

'''+s[end:]
# Several parts of paper previously interpreted batch padding as real generation count.
s=s.replace('''The component experiment also clarifies the relationship between prompt and
completion length. On SciERC, the Mistral $A0\\rightarrow A1$ transition reduces
mean output length from approximately 660 to 370 tokens even though the input
prompt becomes longer. Qwen shows a corresponding reduction from approximately
615 to 385 output tokens. Additional prompt material can therefore reduce decoding
work, which explains why end-to-end latency does not follow input-token count
monotonically.''','''%% revised - reviewer #1 - comments 3,13 - Remove statements equating padded tensor widths with actual decoding work.
The original log suggests that batched generation tensor width changes across
conditions, but such widths do not consistently measure per-response generation
length after EOS/padding. A claim that adding a prompt component reduces actual
output decoding work cannot be based on these counters alone. The independently
retokenized visible completions in Appendix~\\ref{app:expanded-r1} provide a
more interpretable (though approximate) output-length comparison.''')
s=s.replace('''The recorded output-limit-hit flag is 1.000 for the displayed Gemma P6 and
A5 conditions on all three datasets; Mistral and Qwen show more variable
limit-hit frequencies. This does not prove a response was semantically cut off
without inspecting the raw completion, but means differences in F1 and schema
validity cannot be separated fully from decoding-budget effects in the present
experiment. Several ablations increase prompt tokens while
substantially reducing output length. This explains why more elaborate prompts can
occasionally have shorter observed wall-clock time.''','''%% revised - reviewer #1 - comments 3,13 - Correct padding-based truncation artifact.
The original cap flag is based on padded batch tensor width; it cannot
establish individual response truncation, including the many Gemma records
marked 1.000. Since the original per-item EOS/length-stop flags were not
saved, possible decoding-budget effects cannot be distinguished conclusively
from extraction errors. Input token counts and measured latency remain valid
as recorded, but output-length claims require the supplementary retokenized
visible-completion audit.''')
s=s.replace('''$L$ is the recorded output-limit-hit rate.''',r'''$L$ is the \emph{legacy batch-width cap flag}, not verified per-output truncation.''')
s=s.replace('''``At output cap'' is the logged output-limit-hit flag.''','''``At output cap'' is the legacy batch-width flag, not confirmed per-sample truncation.''')
s=s.replace('''mean output tokens''','''mean logged padded width''')
s=s.replace('''Mean output tokens''','''Mean padded width''')
s=s.replace('''mean output length''','''mean padded generation width''')
# Clarify remaining Phase-I historical P6 mismatch that cannot be repaired from missing logs.
s=s.replace('''The original prompt templates are retained below because they define the central
experimental variable of Phase~I.''','''%% revised - reviewer #1 - comment 2 - Historical pilot P6 displayed as an incomplete archived template, not a reconstructed actual long prompt.
The available Phase~I prompt listings are the surviving textual templates.
Their printed P6 lacks the demonstrations and field definitions needed to
explain the apparent 4,200--4,400-token P6 trace in the original Figure~5.
Therefore the exact dispatched Phase~I P6 cannot be reconstructed from this
file; the printed template must not be represented as a verified full request.
The Phase~II P6 is a distinct benchmark-specific construct, not a replicated
literal Phase~I P6. The original listings are retained as archival evidence.''')
# Reviewer 1 comment 14: scientific scope of CLIP text tower as a drift probe.
s=s.replace('''PSI also depends on the semantic representation used to compare fields. Phase~I
showed that its qualitative ordering was preserved between two OpenAI embedding
models, while Phase~II fixes a CLIP text encoder for reproducibility
\\cite{radford2021clip}.''','''%% revised - reviewer #1 - comment 14 - CLIP is a text-change probe, not a scientific-domain encoder.
PSI depends on the embedding representation and is not itself an accuracy
measure. CLIP ViT-B/32 supplies normalized text vectors for short entity spans
and serialized relation strings; the analysis uses only its text encoder.
The model is a fixed reproducible general-purpose semantic-change probe, not
a scientific-domain NER model or an expert verifier. Because CLIP was trained
on image--caption pairs, it may underrepresent technical distinctions,
acronyms and near-synonyms. We therefore restrict conclusions to this
CLIP-specific representation and recommend independent scientific text-only
sentence encoders for replication \\cite{radford2021clip}.''')
# Figure 1 corrected only if PDF patched on server, original retained in source folder otherwise.
# Correct Figure 5 undefined notation without inventing values.
s=s.replace('''Each panel corresponds to one of the six original prompt
    conditions.''','''Each panel corresponds to one of the six original prompt
    conditions. The original panel-specific scales and abbreviated legend remain
    visible for historical transparency; no quantitative cross-panel comparison
    is inferred until original plotting data are reconciled.''')
# Add definitions and same-condition policy; don't fake output.

start=s.index(r'\section{Unresolved validation analyses and interpretation boundaries}')
end=s.index(r'\bibliography{sample}',start)
s=s[:start]+r'''%% revised - reviewer #1 - comments 1,2,8,10,11,12,14 - Explicitly distinguish audited results from tests not yet performed.
\\section{Validation scope and remaining historical constraints}
\\label{app:audit}
The appended Phase~II tables are computed from retained per-document records,
without treating A0/A1 aliases as independent generations. These calculations
settle the available paired comparisons, conditional validity/cost summaries,
relation F1, source-unsupported spans, IDR and clustered PSI--Jaccard diagnostics.
A traceable correction of the \emph{historical} Phase~I experiment additionally
requires its original dispatched P6 text, per-prompt response logs and ECDF source
arrays; these are not included with the released Phase~II data. The original
Figure~1 PDF is preserved but the label ``instruction failure'' is a legacy
annotation: IDR denotes observed field disappearance, not demonstrated error.
The original Figure~5 likewise cannot serve as calibrated quantitative evidence
until its axis scales and Q33/Q66 and $s,m,l$ legend symbols are recovered from
its plotting code. No Phase~I JSON-validity rates are invented.

Same-prompt local repeatability, alternative demonstration selection, and an
independent text-only scientific-span encoder are useful extensions but must
not be presented as empirical findings until they are run and their raw outputs
are archived. Inferences involving legacy output-length and per-document
truncation rates must likewise be limited by the batch-width bookkeeping issue.
The supplementary construction tool verifies what can be reconstructed from
archived data; missing inputs remain reported as missing rather than silently
imputed.

'''+s[end:]

# Repair references to the legacy cap flag in captions, abstract, and RQ5 definitions.
s=s.replace('truncation-at-limit indicators', 'legacy batch-width cap indicators')
s=s.replace('the recorded output-limit-hit rate', 'the recorded batch-width cap flag')
s=s.replace('output-limit-hit proportion', 'legacy batch-width cap proportion')
s=s.replace('a larger output-limit-hit rate', 'a larger legacy batch-width cap flag')
s=s.replace('At output cap', 'Legacy batch cap')
s=s.replace('P6 and A5 conditions on all three datasets', 'P6 and A5 conditions on all three datasets')
# Avoid claiming that an estimate proves per-response termination; label legacy column explicitly.
s=s.replace('mean output-token', 'mean padded-tensor-width')

# Render a stronger standalone starting point offline without inventing missing metrics.
if '--tex-only' in sys.argv:
 for _cmd in ('texttt','ref','section','label','emph','begin','end'):
  s=s.replace('\\\\'+_cmd,'\\'+_cmd)
 s=s.replace('100\\\\%', '100\\%')
 s=s.replace('\\\\_', '\\_')
 s=s.replace(r'Appendix~\ref{app:expanded-r1}', 'the supplementary condition audit')
 s=s.replace(r'Appendix~\ref{app:literal-demos}', 'the original demonstration manifest')
 target=ROOT/'PromptStressLab_R1_ROUND2_SEED.tex'
 target.write_text(s,encoding='utf8')
 print('[WROTE] standalone draft',target, 'bytes',target.stat().st_size)
 raise SystemExit(0)

MET=ROOT/'outputs/metrics/job_metrics_with_aliases.csv'
PSI=ROOT/'outputs/statistics/psi_pairwise.csv'
GEN=ROOT/'outputs/generations'
OUT=ROOT/'outputs/statistics/reviewer1_round2'
OUT.mkdir(parents=True, exist_ok=True)
for path in (MET,PSI,ROOT/'manifests/demo_manifest.json'):
 if not path.exists(): print('[MISSING]',path,flush=True)
assert MET.exists(), 'Need metrics CSV; do not invent missing statistics'
assert (ROOT/'manifests/demo_manifest.json').exists(), 'Need original exact demonstration manifest'
D=pd.read_csv(MET)
D=D[D['condition'].isin([f'P{i}' for i in range(1,7)]+[f'A{i}' for i in range(7)])].copy()
if 'physical_or_alias' in D:
 D=D[(D['condition'].str.startswith('A'))|(D['physical_or_alias']=='physical')]
D.loc[D.dataset=='EBM-NLP','relation_exact_f1']=np.nan
cols=['entity_partial_f1','entity_exact_f1','relation_exact_f1','schema_valid','json_valid','hallucinated_entity_rate','prompt_tokens','output_tokens','seconds_per_item_in_batch','truncated_at_limit']
missing=[c for c in cols if c not in D.columns]
assert not missing, f'Missing data columns: {missing}'
summary=D.groupby(['model_id','dataset','condition'],as_index=False)[cols].mean()
summary['n']=D.groupby(['model_id','dataset','condition']).size().to_numpy()
condition_order={c:i for i,c in enumerate([f'P{i}' for i in range(1,7)]+[f'A{i}' for i in range(7)])}
summary['ord']=summary.condition.map(condition_order)
summary=summary.sort_values(['model_id','dataset','ord'])
# Approximate visible-output counts with actual tokenizer only, never substitute tensor width as length.
original=[]
for f in sorted(GEN.glob('*/predictions.jsonl')):
 with f.open(encoding='utf-8') as h:
  for line in h:
   try: row=json.loads(line)
   except (ValueError,TypeError):continue
   if row.get('condition','').startswith(('P','A')):
    original.append((row.get('model_id'),row.get('dataset'),str(row.get('record_id')),row.get('condition'),row.get('raw_output','')))
R=pd.DataFrame(original,columns=['model_id','dataset','record_id','condition','raw_output']) if original else None
model_cfg=json.loads((ROOT/'config/experiment.json').read_text()) if (ROOT/'config/experiment.json').exists() else {'models':[]}
model_paths={m['model_id']:Path(m['path']) for m in model_cfg['models']}
if R is not None and len(R):
 try:
  from transformers import AutoTokenizer
  length=[]
  for model, data in R.groupby('model_id',sort=False):
   path=model_paths.get(model)
   if not path or not path.exists():
    print('[TOKENIZER NOT FOUND]',model,path,flush=True);continue
   try: tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=True)
   except Exception as e: print('[TOKENIZER NOT LOADABLE]',model,type(e).__name__,str(e)[:180],flush=True);continue
   for idx,row in data.iterrows():
    raw=str(row.raw_output or '')
    n=len(tok.encode(raw,add_special_tokens=False))
    # Since the original runner strips special tokens, re-tokenization is not exact count.
    length.append(dict(model_id=model,dataset=row.dataset,record_id=row.record_id,condition=row.condition,visible_output_tokens_approx=n,raw_empty=int(not raw.strip()),raw_ends_json=int(raw.rstrip().endswith('}'))))
  if length:
   L=pd.DataFrame(length)
   L.to_csv(OUT/'visible_output_token_audit.csv',index=False)
   agg=L.groupby(['model_id','dataset','condition'],as_index=False)[['visible_output_tokens_approx','raw_empty','raw_ends_json']].mean()
   summary=summary.merge(agg,on=['model_id','dataset','condition'],how='left',validate='one_to_one')
 except ImportError:
  print('[SKIP] transformers not installed in active env; no fabricated output length')

# Render robust, self-contained longtables without supplementary files.
def texesc(v):
 return str(v).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def fmt(v,digits=3):
 return '---' if pd.isna(v) else f'{float(v):.{digits}f}'
def table(label,caption,headers,alignment,rows):
 hdr=' & '.join(headers)+r' \\'
 lines=[r'\begingroup',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\setlength{\LTleft}{0pt}',r'\setlength{\LTright}{0pt}',r'\begin{longtable}{'+alignment+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',hdr,r'\midrule',r'\endfirsthead',r'\toprule',hdr,r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot']
 lines += [' & '.join(map(str,x))+r' \\' for x in rows]
 lines += [r'\end{longtable}',r'\endgroup','']
 return '\n'.join(lines)
mn={'qwen3_8b':'Qwen','gemma_3_12b_it':'Gemma','mistral_7b_instruct_v03':'Mistral'}
p0=summary[summary.condition.str.startswith('P')];ab=summary[summary.condition.str.startswith('A')]
section=['%% revised - reviewer #1 - comments 3,4,5,9,13 - Fully reconstructed P1--P6 and A0--A6 condition-by-condition values from existing records.',r'\section{Expanded condition-level model--dataset audit}',r'\label{app:expanded-r1}',
    r'Tables in this section are evaluated from the complete saved metrics, with $n$ denoting source documents per model--dataset--condition. EBM-NLP has no evaluable relation F1. A0 and A1 are reused predictions and are not independent generations.']
rows=[]
for _,r in p0.iterrows():
 rows.append((mn.get(r.model_id,texesc(r.model_id)),texesc(r.dataset),r.condition,int(r.n),fmt(r.entity_partial_f1),fmt(r.entity_exact_f1),fmt(r.relation_exact_f1),fmt(r.schema_valid),fmt(r.json_valid),fmt(r.hallucinated_entity_rate)))
section.append(table('tab:r1-all-p-quality','All main P1--P6 conditions: mean reference quality and structural validity. Hallucination denotes source-nonoccurring predicted spans, not benchmark false positives.',['Model','Data','Prompt','$n$','Partial F1','Exact F1','Relation F1','Schema','JSON','Unsupported'], 'lllrrrrrrr',rows))
rows=[]
for _,r in ab.iterrows():rows.append((mn.get(r.model_id,texesc(r.model_id)),texesc(r.dataset),r.condition,int(r.n),fmt(r.entity_partial_f1),fmt(r.relation_exact_f1),fmt(r.schema_valid),fmt(r.json_valid),fmt(r.hallucinated_entity_rate)))
section.append(table('tab:r1-all-a-quality','All ablations A0--A6: mean reference quality and structural validity, including exact aliases A0 and A1.',['Model','Data','Prompt','$n$','Partial F1','Relation F1','Schema','JSON','Unsupported'], 'lllrrrrrr',rows))
rows=[]
for _,r in p0.iterrows():
 rows.append((mn.get(r.model_id,texesc(r.model_id)),texesc(r.dataset),r.condition,int(r.n),fmt(r.prompt_tokens,1),fmt(r.output_tokens,1),fmt(r.seconds_per_item_in_batch,2),fmt(r.truncated_at_limit)))
section.append(table('tab:r1-all-p-cost',r"All main P1--P6 cost and execution logs. ``Padded width'' and ``batch cap'' are \emph{legacy batched-tensor} measurements; not actual per-response output token counts or truncation rates.",['Model','Data','Prompt','$n$','Input tokens','Padded width','s/item','Batch cap'], 'lllrrrrr',rows))
rows=[]
for _,r in ab.iterrows():
 rows.append((mn.get(r.model_id,texesc(r.model_id)),texesc(r.dataset),r.condition,int(r.n),fmt(r.prompt_tokens,1),fmt(r.output_tokens,1),fmt(r.seconds_per_item_in_batch,2),fmt(r.truncated_at_limit)))
section.append(table('tab:r1-all-a-cost','All A0--A6 cost and execution logs (same batch-width measurement limitation as Table~\\ref{tab:r1-all-p-cost}).',['Model','Data','Prompt','$n$','Input tokens','Padded width','s/item','Batch cap'], 'lllrrrrr',rows))
if 'visible_output_tokens_approx' in summary:
 section.append(r'%% revised - reviewer #1 - comments 3,13 - Retokenized visible completions (not padded generated tensor width).')
 rows=[]
 for _,r in summary.iterrows():
  if not pd.isna(r.visible_output_tokens_approx):
   rows.append((mn.get(r.model_id,texesc(r.model_id)),texesc(r.dataset),r.condition,int(r.n),fmt(r.visible_output_tokens_approx,1),fmt(r.raw_ends_json),fmt(r.raw_empty)))
 section.append(table('tab:r1-visible-length','Offline re-tokenization of saved raw completions with the original tokenizer. Values estimate visible output length but omit special tokens and cannot determine whether EOS or token limit caused termination. ``Ends JSON'' is a surface suffix check, not full parsing.',['Model','Data','Prompt','$n$','Visible tokens','Ends JSON','Empty'], 'lllrrrr',rows))
else:
 section.append(r'No retokenized output-length table is asserted, because access to the original local tokenizer files is required to compute it accurately.')

manifest=ROOT/'manifests/demo_manifest.json'
if manifest.exists():
 demos=json.loads(manifest.read_text())['datasets']
 section += [r'%% revised - reviewer #1 - comments 9,12 - Exact unabridged training-split demonstrations from the checked manifest.',r'\section{All fixed few-shot demonstrations (verbatim)}',r'\label{app:literal-demos}',r'The following blocks reproduce all three original training-partition input/output pairs per dataset in prompt order. Their identity is fixed across every evaluation document; alternative-example robustness was not tested.']
 for ds, examples in demos.items():
  for k,e in enumerate(examples,1):
   section.append(f'\\paragraph{{{texesc(ds)} demonstration {k}: {texesc(e["record_id"])}}}')
   section.append(r'\begin{lstlisting}[style=promptstyle]')
   section.append('INPUT:\n'+e['input_text']+'\nOUTPUT:\n'+json.dumps(e['output'],ensure_ascii=False,separators=(',',':')))
   section.append(r'\end{lstlisting}')
else:
 section += [r'The complete nine worked demonstrations cannot be printed without manifests/demo_manifest.json; the published IDs in Table~\ref{tab:demo-ids} remain the authoritative source identifiers.']

# Optional targeted-control evidence: only append results actually saved by a real experiment.
controls_file=OUT/'targeted_controls/targeted_controls_summary.json'
if controls_file.exists():
    c=json.loads(controls_file.read_text(encoding='utf8'))
    section += [r'%% revised - reviewer #1 - comments 8,12,14 - Observed small control-run results; no unmeasured endpoint filled.',
                r'\section{Targeted local reproducibility controls}',r'\label{app:targeted-controls}',
                r'The following are separate post-hoc local controls, not part of the original 8,526 physical generations. Document identifiers and new predictions are archived with the control-run script. The one-model, small-subset scope limits external generalization, and original batch scheduling differs from these single-item local reruns.']
    rows=[]
    if c.get('noise_documents',0)>0:
        rows.append(('Qwen SciERC P3, two exact repeats',c['noise_documents'],fmt(c.get('same_condition_jaccard_mean')),fmt(c.get('same_condition_clip_psi_mean')),fmt(c.get('same_condition_clip_psi_p95'))))
    if rows:
        section.append(table('tab:r1-noise-floor',r'Same-prompt run-to-run noise floor, identical inputs and greedy decoding. CLIP PSI entries are unavailable unless the original local CLIP encoder was explicitly supplied.', ['Setting','$n$','Jaccard mean','CLIP PSI mean','CLIP PSI P95'],'lrrrr',rows))
    if c.get('same_document_comparisons',0)>0:
        section.append(r'On the same '+str(int(c['same_document_comparisons']))+
            r' SciERC documents, original P3--P4 prompt-change CLIP PSI averaged $'+
            fmt(c.get('prompt_change_psi_same_docs_mean'),4)+r'$, compared with a repeated-P3 baseline of $'+
            fmt(c.get('same_condition_clip_psi_mean'),4)+r'$. Prompt-change PSI exceeded corresponding same-prompt noise in $'+
            fmt(100*c.get('prompt_change_above_noise_fraction',0),1)+r'\%$ of tested documents. Original P3--P4 records used a different batch scheduling pattern from the new one-item repeats.')
    if c.get('alternative_demo_documents',0)>0:
        n=int(c['alternative_demo_documents'])
        dlt=fmt(c.get('alternative_demo_mean_delta_f1'),4)
        pval=fmt(c.get('alternative_demo_wilcoxon_p'),4)
        section.append(r'An independent fixed training-split demonstration triplet was evaluated on '+str(n)+r' EBM-NLP documents using Qwen3-8B. Relative to the original P5 demonstration set, mean paired partial-entity F1 difference was $'+dlt+r'$ (unadjusted two-sided Wilcoxon $p='+pval+r'$). The experimental unit is the document. This probes example identity, but does not replace broader multi-seed demonstration sampling.')
    if c.get('text_encoder_n_transitions',0)>0:
        section.append(r'An independent frozen text-only encoder was applied to '+str(int(c['text_encoder_n_transitions']))+r' Qwen--SciERC adjacent main-prompt transitions, obtaining Spearman $\rho='+fmt(c.get('text_encoder_spearman_psi_abs_delta_f1'),4)+r'$ with absolute partial-F1 changes. Encoder comparisons are descriptive unless tested on exactly the same subset with document clustering.')
    if not rows and not c.get('alternative_demo_documents',0) and not c.get('text_encoder_n_transitions',0):
        section.append(r'The control-run summary contains no completed experiment; no result is asserted.')

# Optional surgical vector-PDF label correction: never modify original source figure.
try:
    import fitz
    candidates=[ROOT/'figs/Prompt Optimization Experimental Pipeline2.pdf', ROOT/'figs/Prompt Optimization Experimental Pipeline2_reviewer_fixed.pdf']
    source=candidates[0]
    if source.is_file():
        doc=fitz.open(source)
        fixed=0
        for pg in doc:
            for needle in ('instruction failure','Instruction failure','INSTRUCTION FAILURE'):
                boxes=pg.search_for(needle)
                for rect in boxes:
                    # White-fill only the exact text bounding box, then label with its corrected meaning.
                    pg.add_redact_annot(rect,fill=(1,1,1))
                    pg.apply_redactions(images=0,graphics=0,text=0)
                    pg.insert_text((rect.x0, rect.y1-1), 'field disappearance', fontsize=min(8, max(5,rect.height*0.67)),color=(0,0,0))
                    fixed+=1
        if fixed:
            dest=ROOT/'figs/Prompt Optimization Experimental Pipeline2_reviewer_fixed.pdf'
            doc.save(dest,garbage=4,deflate=True)
            s=s.replace('figs/Prompt Optimization Experimental Pipeline2.pdf','figs/Prompt Optimization Experimental Pipeline2_reviewer_fixed.pdf',1)
            print('[FIG1 CORRECTED] vector-label replacements:',fixed,flush=True)
        else:
            print('[FIG1 UNALTERED] Label was not selectable PDF text; revise its vector source / editable master.',flush=True)
        doc.close()
except ImportError:
    print('[FIG1] PyMuPDF unavailable; source image not changed',flush=True)
except Exception as exc:
    print('[FIG1] no safe alteration:',type(exc).__name__,str(exc)[:150],flush=True)

# Normalize newly added command spelling in raw LaTeX literal snippets.
for _cmd in ('texttt','ref','section','label','emph','begin','end'):
    s=s.replace('\\\\'+_cmd,'\\'+_cmd)
s=s.replace('100\\\\%', '100\\%')
s=s.replace('\\\\_', '\\_')

# Insert BEFORE bibliography so standalone TeX remains single-file.
newsec='\n'.join(section)+'\n'
s=s.replace(r'\bibliography{sample}',newsec+r'\bibliography{sample}',1)
s=s.replace('''The aggregate console extract does not contain every P1--P6 condition-level
measurement, so the additional full-condition table is not asserted here.''','''The complete P1--P6 condition-level summary is now given in
Appendix~\\ref{app:expanded-r1}, reconstructed from the original record-level CSV.''')
s=s.replace('''condition-level appendix must not be
subtracted directly from the 84-document A5 mean''','''condition-level appendix must not be
subtracted directly from the 84-document A5 mean''')
# Correct references about literally missing demos only when all demos embedded.
if manifest.exists():
 s=s.replace('''The remaining seven source strings are not reconstructed from aggregate
evaluation results.''','''The remaining seven exact pairs, together with the first two, are reproduced
verbatim in Appendix~\\ref{app:literal-demos} from the original manifest.''')

result=ROOT/'PromptStressLab_R1_ROUND2_COMPLETE_INLINE.tex'
result.write_text(s,encoding='utf8')
assert r'\input{reviewer1_posthoc_tables.tex}' not in s
assert s.count(r'\begin{figure')==s.count(r'\end{figure')
assert s.count(r'\begin{table')==s.count(r'\end{table')
print('[WROTE]',result,'bytes',result.stat().st_size,'main rows',len(p0),'ablation rows',len(ab))
print('[CHECK] fig environments:',s.count(r'\begin{figure'))
print('[CHECK] citations preserved, no additional input files')
print('[IMPORTANT] Phase I source logs and 3 targeted validations are NOT auto-fabricated.')
