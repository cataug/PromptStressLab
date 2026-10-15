# October 2026 revision snapshot

This directory records the October 2026 revision of
**Prompt Composition and Length Trade-offs in Structured
Information Extraction with Large Language Models**.

It is separate from the original experiment and the
September post-hoc analysis.

## Contents

- `manuscript/` — LaTeX, references and article figures.
- `code/` — revision, statistical and plotting scripts.
- `results/phase1/` — recomputed Phase I metrics and figures.
- `results/phase2/` — available post-hoc Phase II results.
- `historical_reference/` — original numerical workbooks.
- `MANIFEST.json` — file origins and SHA-256 checksums.

## Phase I

The original study contains 3,180 independent API generations
for 530 abstracts and six prompt conditions.

The 3,180 parsed JSON files are derivatives of those generations.

The historical P6 prompt constructor was recovered and contains
30 repetitions of its rule block.

The corrected reanalysis uses TF-IDF and explicit handling of
empty-to-empty, empty-to-nonempty, and nonempty-to-empty fields.

The original outputs contain no independently recorded
same-condition repeated calls. Therefore, the run-to-run noise
floor cannot be estimated from this archive.

The original raw JSON archive is not duplicated here.

## Phase II

The existing repository retains the baseline multi-model
experimental pipeline and original results.

The October analysis adds post-hoc paired inference,
component comparisons, construct validation and corrections
to interpretation.

The legacy batch-padded output-token counter cannot establish
true per-sequence truncation without an additional stopping audit.

## Important limitations

The historical JSON-validity flags need to be interpreted
alongside parser-recovery metadata.

An archived ACL transfer experiment cannot be fully verified
from the 530-document Phase I archive alone.

This snapshot documents work in progress and does not claim
that every reviewer request has already been resolved.

## Historical Phase I implementation

https://github.com/catauggie/PromptEng26
