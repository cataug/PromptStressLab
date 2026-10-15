#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

SRC = Path("/home/tahiti/PromptStressLab")
CHECKOUT = Path("/home/tahiti/PromptStressLab_october_git")

REMOTE = "git@github.com:cataug/PromptStressLab.git"
FOLDER = "october_2026_revision"
MAIN = "PromptStressLab_PHASE1_FINAL_INLINE.tex"

MAX_FILE = 25 * 1024 * 1024
MAX_TOTAL = 150 * 1024 * 1024

EXCLUDE_DIRS = {
    "generations", "predictions", "raw", "raw_outputs",
    "data", "datasets", "models", "checkpoints",
    "cache", ".cache", "__pycache__", ".git",
    ".venv", "venv", "phase1_original",
    "legacy_PromptEng26", "phase1_reanalysis_2026",
    "before_viridis", "node_modules",
}

ALLOWED_RESULTS = {
    ".csv", ".tsv", ".tex", ".json",
    ".xlsx", ".pdf", ".png", ".md",
}

SECRET = re.compile(
    rb"(?:sk-[a-zA-Z0-9_-]{18,}|"
    rb"ghp_[A-Za-z0-9]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{30,}|"
    rb"AKIA[A-Z0-9]{16}|"
    rb"hf_[a-zA-Z0-9]{28,})"
)

GIT_ENV = dict(
    os.environ,
    GIT_SSH_COMMAND=(
        "ssh -o BatchMode=yes "
        "-o ConnectTimeout=15 "
        "-o StrictHostKeyChecking=accept-new"
    ),
)

entries = []
warnings = []
bytes_copied = 0


def run(*args, cwd=None, check=True, quiet=False):
    print("+", " ".join(map(str, args)), flush=True)
    return subprocess.run(
        args,
        cwd=cwd,
        env=GIT_ENV,
        check=check,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE if quiet else None,
        text=True,
    )


def add(src, rel, required=False):
    global bytes_copied

    src = Path(src)

    if not src.is_file():
        if required:
            raise RuntimeError(f"Required input not found: {src}")
        warnings.append(f"Not found: {src}")
        return

    if src.is_symlink():
        warnings.append(f"Symlink not copied: {src}")
        return

    size = src.stat().st_size

    if size > MAX_FILE:
        warnings.append(f"File too large, skipped: {src}")
        return

    if bytes_copied + size > MAX_TOTAL:
        raise RuntimeError(
            "150 MiB safety limit reached; inspect selection"
        )

    text_extensions = {
        ".py", ".sh", ".tex", ".bib", ".csv",
        ".tsv", ".json", ".md", ".cls",
        ".sty", ".bst", ".txt",
    }

    if src.suffix.lower() in text_extensions:
        if SECRET.search(src.read_bytes()):
            raise RuntimeError(
                f"Possible embedded credential detected: {src}"
            )

    dst = CHECKOUT / FOLDER / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)

    sha = hashlib.sha256(dst.read_bytes()).hexdigest()

    entries.append({
        "source": (
            str(src.relative_to(SRC))
            if src.is_relative_to(SRC)
            else str(src)
        ),
        "destination": rel.as_posix(),
        "bytes": size,
        "sha256": sha,
    })

    bytes_copied += size


def walk_results(directory, relative_root, mode):
    if not directory.is_dir():
        return

    for base, dirs, names in os.walk(directory):
        dirs[:] = sorted(
            d for d in dirs
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        )

        for name in sorted(names):
            p = Path(base) / name

            if p.suffix.lower() not in ALLOWED_RESULTS:
                continue

            sub = p.relative_to(directory)

            if mode == "phase2":
                lower = str(sub).lower()

                selected = any(
                    term in lower
                    for term in (
                        "posthoc",
                        "reviewer",
                        "round2",
                        "ablation",
                        "per_condition",
                        "p6_vs_a5",
                        "phase2_idr",
                        "psi_jaccard",
                        "rq5",
                        "mcnemar",
                        "bootstrap",
                        "paired_inference",
                    )
                )

                if not selected:
                    continue

            add(p, relative_root / sub)


def main():
    print("=" * 85)
    print("PROMPTSTRESSLAB — OCTOBER 2026 GITHUB PUBLICATION")
    print("=" * 85)

    if not SRC.is_dir():
        raise RuntimeError(f"Project directory missing: {SRC}")

    maintex = SRC / MAIN

    if not maintex.is_file():
        raise RuntimeError(f"Missing LaTeX manuscript: {maintex}")

    # -------------------------------------------------------
    # 1. SSH
    # -------------------------------------------------------

    print("\n[1/7] VERIFY SSH", flush=True)

    run("git", "ls-remote", REMOTE, "HEAD")

    # -------------------------------------------------------
    # 2. CLONE OR UPDATE
    # -------------------------------------------------------

    print("\n[2/7] PREPARE GITHUB CHECKOUT", flush=True)

    if (CHECKOUT / ".git").is_dir():

        origin = run(
            "git", "remote", "get-url", "origin",
            cwd=CHECKOUT,
            quiet=True,
        ).stdout.strip()

        if origin != REMOTE:
            raise RuntimeError(
                f"Existing checkout has another remote: {origin}"
            )

        status = run(
            "git", "status", "--porcelain",
            cwd=CHECKOUT,
            quiet=True,
        ).stdout.strip()

        if status:
            raise RuntimeError(
                "Existing checkout has uncommitted changes: "
                + status[:800]
            )

        run("git", "checkout", "main", cwd=CHECKOUT)
        run(
            "git", "pull", "--ff-only", "origin", "main",
            cwd=CHECKOUT,
        )

    else:
        if CHECKOUT.exists() and any(CHECKOUT.iterdir()):
            raise RuntimeError(
                f"Checkout destination already occupied: {CHECKOUT}"
            )

        run("git", "clone", REMOTE, str(CHECKOUT))
        run("git", "checkout", "main", cwd=CHECKOUT)

    root = CHECKOUT / FOLDER
    root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 3. MANUSCRIPT, BIBLIOGRAPHY AND FIGURES
    # -------------------------------------------------------

    print("\n[3/7] COLLECT MANUSCRIPT AND FIGURES", flush=True)

    add(
        maintex,
        Path("manuscript") / MAIN,
        required=True,
    )

    for ext in ("*.bib", "*.cls", "*.sty", "*.bst"):
        for p in sorted(SRC.glob(ext)):
            add(p, Path("manuscript") / p.name)

    main_text = maintex.read_text(
        encoding="utf-8",
        errors="replace",
    )

    refs = re.findall(
        r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}",
        main_text,
    )

    for name in sorted(set(refs)):
        name = name.strip()

        rel = Path(name)

        if rel.is_absolute() or ".." in rel.parts:
            warnings.append(
                f"Unsafe graphic reference skipped: {name}"
            )
            continue

        if rel.suffix:
            candidates = [SRC / rel]
        else:
            candidates = [
                SRC / (name + ext)
                for ext in (".pdf", ".png", ".jpg")
            ]

        found = next(
            (p for p in candidates if p.is_file()),
            None,
        )

        if found:
            add(
                found,
                Path("manuscript") / found.relative_to(SRC),
            )
        else:
            warnings.append(
                f"Manuscript graphic missing: {name}"
            )

    # -------------------------------------------------------
    # 4. PYTHON AND SHELL CODE
    # -------------------------------------------------------

    print("\n[4/7] COLLECT REVISION CODE", flush=True)

    for ext in ("*.py", "*.sh"):
        for p in sorted(SRC.glob(ext)):
            add(p, Path("code") / "root" / p.name)

    for dirname in (
        "scripts",
        "reviewer1_round2",
        "reviewer_posthoc_20260926",
    ):
        sd = SRC / dirname

        if not sd.is_dir():
            continue

        for p in sorted(sd.rglob("*")):
            if (
                p.is_file()
                and p.suffix.lower() in {".py", ".sh"}
            ):
                add(
                    p,
                    Path("code") / dirname / p.relative_to(sd),
                )

    # -------------------------------------------------------
    # 5. OCTOBER RESULTS
    # -------------------------------------------------------

    print("\n[5/7] COLLECT RESULTS", flush=True)

    walk_results(
        SRC / "outputs" / "phase1_reanalysis_2026",
        Path("results") / "phase1",
        "phase1",
    )

    walk_results(
        SRC / "outputs",
        Path("results") / "phase2",
        "phase2",
    )

    for p in sorted(SRC.glob("*")):
        if (
            p.is_file()
            and p.suffix.lower() in {".csv", ".tsv"}
        ):
            add(
                p,
                Path("results") / "root_tables" / p.name,
            )

    # Historical numeric workbooks (without raw abstracts).
    for name in (
        "run_summary.xlsx",
        "stability_metrics.xlsx",
        "json_metrics.xlsx",
    ):
        p = SRC / "legacy_PromptEng26" / name

        if p.is_file():
            add(
                p,
                Path("historical_reference") / name,
            )

    if not any(
        e["destination"].startswith("results/phase1/")
        for e in entries
    ):
        raise RuntimeError(
            "Phase I results missing: refusing incomplete snapshot"
        )

    for stem in (
        "Figure3_PSI_ECDF_with_insets_corrected",
        "Figure4_Field_Volatility_gradient_corrected",
        "Figure5_Pareto_common_axes_local_insets_corrected",
    ):
        p = root / "manuscript" / "figs" / (stem + ".pdf")

        if not p.is_file():
            warnings.append(
                f"Expected Figure 3–5 PDF missing: {stem}"
            )

    # -------------------------------------------------------
    # 6. README AND MANIFEST
    # -------------------------------------------------------

    print("\n[6/7] WRITE README AND MANIFEST", flush=True)

    readme = """# October 2026 revision snapshot

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
"""

    (root / "README.md").write_text(
        readme,
        encoding="utf-8",
    )

    entries.append({
        "source": "generated README",
        "destination": "README.md",
        "bytes": len(readme.encode()),
        "sha256": hashlib.sha256(
            readme.encode()
        ).hexdigest(),
    })

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_root": "local PromptStressLab working directory",
        "github_remote": REMOTE,
        "subdirectory": FOLDER,
        "files": entries,
        "warnings": warnings,
    }

    (root / "MANIFEST.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        f"\nFiles selected: {len(entries)}",
        f"\nTotal size: {bytes_copied / 1024**2:.2f} MiB",
        flush=True,
    )

    for warning in warnings:
        print("[WARN]", warning, flush=True)

    # -------------------------------------------------------
    # 7. GIT ADD, COMMIT, PUSH
    # -------------------------------------------------------

    print("\n[7/7] GIT COMMIT AND SSH PUSH", flush=True)

    run(
        "git", "add", "-f", "--", FOLDER,
        cwd=CHECKOUT,
    )

    run(
        "git", "status", "--short", "--", FOLDER,
        cwd=CHECKOUT,
    )

    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=CHECKOUT,
    ).returncode

    if diff == 0:
        print("[OK] No new changes to commit")

    elif diff == 1:

        run(
            "git", "commit",
            "-m",
            "Add October 2026 Phase I reanalysis and manuscript revision",
            cwd=CHECKOUT,
        )

        run(
            "git", "push", "origin", "main",
            cwd=CHECKOUT,
        )

    else:
        raise RuntimeError("git diff --cached failed")

    print("\n========== FINAL GIT STATUS ==========")

    run(
        "git", "log", "-1", "--oneline",
        cwd=CHECKOUT,
    )

    print("\n[OK] OCTOBER REVISION PUBLISHED")
    print(
        "https://github.com/cataug/PromptStressLab/"
        "tree/main/october_2026_revision"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "[ERROR]",
            type(exc).__name__ + ":",
            exc,
            flush=True,
        )
        sys.exit(1)
