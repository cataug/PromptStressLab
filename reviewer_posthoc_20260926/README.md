# Reviewer post-hoc analyses — 2026-09-26

Post-hoc analyses added in response to reviewer requests for stronger
validation of prompt sensitivity and the proposed PSI measure.

## Included

### Multi-encoder PSI robustness
- CLIP ViT-B/32
- MPNet
- SPECTER
- set-Jaccard and token-Jaccard controls
- document-level, transition-rank, and field-rank agreement

### PSI and extraction-quality change
- Spearman association with signed and absolute F1 changes
- model-level and dataset-level analyses
- practical risk-screening analysis using AUROC/AUPRC

### ProSA-style comparison
- performance-sensitivity comparison derived from existing evaluation results

### POSIX
- exact teacher-forced POSIX from the original saved model responses
- original P1--P6 prompts reconstructed with the production prompt builder
- prompt reconstruction independently verified for all 5,526 main jobs by:
  - SHA256 equality
  - exact character-count equality
- no new LLM generations were required
- 33,156 teacher-forced prompt/response likelihood evaluations
- 921 document-model cases
- 30 cross-prompt comparisons per document-model case

### Adjacent POSIX
- matched P1->P2, P2->P3, P3->P4, P4->P5, P5->P6 comparisons
- 4,605 adjacent transitions
- direct apples-to-apples comparison with adjacent PSI
- association with absolute partial-F1 change

## Main headline results

- Semantic-encoder document-level PSI agreement is very high
  (Spearman rho approximately 0.973--0.988).
- Semantic PSI is strongly associated with absolute extraction-quality changes.
- PSI provides useful screening ability for large F1 changes.
- Adjacent POSIX and PSI are moderately positively associated when evaluated
  on exactly the same prompt transitions.
- Adjacent POSIX is also associated with extraction-quality change, but the
  semantic PSI variants show stronger associations on this benchmark.
- Canonical all-pairs POSIX, adjacent PSI, and ProSA-style performance
  sensitivity therefore capture related but non-identical aspects of prompt
  dependence.

## Excluded from Git snapshot

Large/intermediate artifacts are intentionally omitted, including:
- reconstructed_main_prompts.jsonl
- posix_pair_logprobs.csv
- model weights
- raw experiment generations/predictions

These remain available in the local experiment workspace.

## Directories

- `scripts/` — analysis and figure-generation code
- `stats/` — lightweight CSV/TEX/JSON/TXT statistics
- `figures/` — PDF and PNG post-hoc figures
