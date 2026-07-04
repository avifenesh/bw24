#!/usr/bin/env python3
"""dataset.py — loader, validator, feature extraction, and leave-one-out splitter
for the bw24 optimization corpus (research/tune-data/*.jsonl).

This is the shared data layer for the training arm (arm 2 of bw24): every model
we ever build — retrieval baseline now, gradient-boosted head at ~200 records,
LoRA judge at ~1000 — reads records through THIS module so features and the
leave-one-out eval protocol stay identical and comparable across model
generations.

Zero third-party dependencies (stdlib only) so it runs under any Python on the
box, including the plain system interpreter. baseline.py adds numpy + an
embedder on top of this.

Corpus schema (see research/tune-data/README.md):
    ts, rig, commit, change, kernel, baseline{metric:val}, result{metric:val},
    speedup_graph (float|null), clock_note, accuracy{gate:...}, label, notes

Prediction framing (arm 2 goal):
    INPUTS  = rig, kernel/component, change (free text), baseline metrics
    TARGETS = speedup (signed) and gate-break risk (label / accuracy gates)
    -> result{}, speedup_graph, accuracy{}, label are OUTCOMES. They must never
       leak into the INPUT representation used at query time.

Usage:
    python3 dataset.py                 # validate + print corpus summary
    python3 dataset.py --file path.jsonl
    from dataset import load, leave_one_out, document_text, query_text
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# The corpus is the WHOLE tune-data directory: every per-rig file
# (rig5090.jsonl, g7e-rtx6000.jsonl, l40s-sm89.jsonl, ...) is globbed and
# concatenated so `rig` is a real multi-value conditioning variable rather than
# a constant. DEFAULT_CORPUS is that directory; load() resolves it to all
# *.jsonl inside. A single file or a glob pattern is still accepted.
DEFAULT_CORPUS_DIR = os.path.normpath(os.path.join(HERE, "..", "tune-data"))
DEFAULT_CORPUS = DEFAULT_CORPUS_DIR
# The primary rig's file — the oracle's "append your latest result here" target
# (see ORACLE.md). Named separately so appends still know exactly one file.
PRIMARY_RIG_FILE = os.path.join(DEFAULT_CORPUS_DIR, "rig5090.jsonl")
# Model-quality tracking rows (rig="corpus-meta") are written here by
# train_gbm.py --append-meta so the corpus tracks how the stage-b head scores as
# it grows. These are NOT training data: corpus_files() EXCLUDES this file from
# the default glob so meta rows never enter the LOO folds. Load it explicitly
# (--file model-meta.jsonl) to inspect the history.
META_RIG = "corpus-meta"
MODEL_META_FILE = os.path.join(DEFAULT_CORPUS_DIR, "model-meta.jsonl")

REQUIRED_FIELDS = (
    "ts", "rig", "commit", "change", "kernel",
    "baseline", "result", "speedup_graph", "clock_note",
    "accuracy", "label", "notes",
)
VALID_LABELS = {"positive", "negative", "neutral"}

# Which extracted feature keys are model INPUTS vs TARGETS. The phase-(b)
# gradient-boosted head trains on INPUT_FEATURES -> TARGET_*; keeping the split
# declared here (not in each trainer) is what makes generations comparable.
INPUT_FEATURES = (
    "rig", "component", "is_measurement", "is_revert",
    "n_baseline_metrics", "n_result_metrics", "has_accuracy_gate",
    "change_len", "kernel_len",
)
TARGET_FIELDS = ("target_label", "target_speedup", "target_speedup_sign")

# Tolerance band around 1.0 within which a speedup ratio counts as "flat"
# (neither a win nor a loss). 0.5% mirrors the measurement noise floor the
# corpus records (N=5 spreads of <0.3-0.7%).
SPEEDUP_FLAT_TOL = 0.005


# ---------------------------------------------------------------------------
# Component bucketing (kernel free-text -> coarse component)
# ---------------------------------------------------------------------------
# Ordered rules; first match on lowercased (kernel + " " + change) wins. These
# buckets are the coarse "which part of the engine" axis the autotuner prior
# needs — kernel strings are too high-cardinality to use raw.
_COMPONENT_RULES = (
    ("spec-decode", ("spec", "mtp", "draft", "verify", "nextn", "leviathan", "frspec", "pmin", "p-min")),
    ("flash-attn", ("fa_decode", "fa_prefill", "fa_split", "flashatt", "fa ", "fa_", "attn", "attention", "softmax")),
    ("gemm-mmq", ("mmq", "mul_mat_q", "gemm", "stream-k", "stream_k", "streamk", "mmq_x", "mma")),
    ("matvec", ("mmvq", "matvec", "qmatvec", "mul_mat_vec", "dp4a")),
    ("ssm-conv", ("conv", "ssm", "gdn", "linear_attn", "scan", "ring")),
    ("norm", ("rms_norm", "l2_norm", "rms", "norm", "silu")),
    ("runtime-io", ("cache", "server", "kv", "prime", "embed_gather", "snapshot", "gather", "htod", "dtoh")),
    ("measurement", ("nsys", "ncu", "bench", "harness", "e2e", "probe", "full engine", "all")),
)


def component_of(kernel: str, change: str = "") -> str:
    hay = (str(kernel) + " " + str(change)).lower()
    for name, keys in _COMPONENT_RULES:
        for k in keys:
            if k in hay:
                return name
    return "other"


def _is_measurement(rec: dict) -> bool:
    commit = str(rec.get("commit", "")).lower()
    change = str(rec.get("change", "")).lower()
    if "measurement" in commit:
        return True
    for kw in ("measurement", "probe", "decomposition", "sweep", "interleaved",
               "nsys", "ncu", "parity probe", "a/b protocol", "assessment"):
        if kw in change:
            return True
    # neutral-labelled rows with no headline ratio are almost always protocol rows
    return False


def _is_revert(rec: dict) -> bool:
    commit = str(rec.get("commit", "")).lower()
    change = str(rec.get("change", "")).lower()
    return ("revert" in change or "revert" in commit
            or "reverted" in commit or "reverted" in change)


# ---------------------------------------------------------------------------
# Record model
# ---------------------------------------------------------------------------
@dataclass
class Record:
    idx: int
    raw: dict
    issues: list = field(default_factory=list)
    source: str = ""  # basename of the rig file this record came from

    # convenience accessors -------------------------------------------------
    @property
    def label(self) -> str:
        return self.raw.get("label", "")

    @property
    def rig(self) -> str:
        return self.raw.get("rig", "")

    @property
    def kernel(self) -> str:
        return self.raw.get("kernel", "")

    @property
    def change(self) -> str:
        return self.raw.get("change", "")

    @property
    def notes(self) -> str:
        return self.raw.get("notes", "")

    @property
    def component(self) -> str:
        return component_of(self.kernel, self.change)

    @property
    def signed_speedup(self) -> Optional[float]:
        """Headline speedup ratio (>1 faster, <1 slower). None when the record
        has no single-ratio headline (measurement/protocol rows)."""
        s = self.raw.get("speedup_graph", None)
        if isinstance(s, bool):  # guard: bool is an int subclass
            return None
        if isinstance(s, (int, float)):
            return float(s)
        return None

    @property
    def speedup_sign(self) -> Optional[str]:
        s = self.signed_speedup
        if s is None:
            return None
        if s > 1.0 + SPEEDUP_FLAT_TOL:
            return "up"
        if s < 1.0 - SPEEDUP_FLAT_TOL:
            return "down"
        return "flat"


# ---------------------------------------------------------------------------
# Text representations  (the leakage boundary lives here)
# ---------------------------------------------------------------------------
def document_text(rec: Record) -> str:
    """Rich text used to INDEX a past record. Includes notes because a stored
    precedent's mechanism/lesson is exactly what makes it a useful retrieval
    hit and explanation. Corpus-side only — never used to represent a fresh,
    not-yet-run proposal."""
    return (
        f"component: {rec.component}. rig: {rec.rig}. "
        f"change: {rec.change} "
        f"kernel: {rec.kernel} "
        f"notes: {rec.notes}"
    )


def query_text(rec: Record) -> str:
    """Text used to REPRESENT a proposal at query time. change + kernel only —
    this is all a grind session has before running the experiment. Used both by
    the live oracle and by leave-one-out eval so the eval measures the oracle's
    real operating conditions (no notes/result/label leakage)."""
    return f"component: {rec.component}. change: {rec.change} kernel: {rec.kernel}"


def proposal_query_text(change: str, kernel: str = "") -> str:
    """Same shape as query_text() but for an ad-hoc proposal string typed by a
    grind session (no Record yet)."""
    comp = component_of(kernel, change)
    return f"component: {comp}. change: {change} kernel: {kernel}"


# ---------------------------------------------------------------------------
# Structured feature extraction
# ---------------------------------------------------------------------------
def record_to_features(rec: Record) -> dict:
    """Flat feature dict for the future gradient-boosted head. INPUT_FEATURES
    are safe to condition on at proposal time; TARGET_FIELDS are outcomes. The
    split is declared at module top so every trainer uses the same one."""
    baseline = rec.raw.get("baseline", {}) or {}
    result = rec.raw.get("result", {}) or {}
    accuracy = rec.raw.get("accuracy", {}) or {}
    feats = {
        # ---- INPUTS ----
        "rig": rec.rig,
        "component": rec.component,
        "is_measurement": _is_measurement(rec.raw),
        "is_revert": _is_revert(rec.raw),
        "n_baseline_metrics": len(baseline) if isinstance(baseline, dict) else 0,
        "n_result_metrics": len(result) if isinstance(result, dict) else 0,
        "has_accuracy_gate": bool(accuracy),
        "change_len": len(rec.change),
        "kernel_len": len(rec.kernel),
        # ---- TARGETS ----
        "target_label": rec.label,
        "target_speedup": rec.signed_speedup,
        "target_speedup_sign": rec.speedup_sign,
    }
    return feats


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_record(rec: Record) -> list:
    """Return a list of human-readable issues (empty == clean). Non-fatal: the
    corpus is hand-authored, so we warn and keep going rather than reject."""
    issues = []
    r = rec.raw
    for f in REQUIRED_FIELDS:
        if f not in r:
            issues.append(f"missing field '{f}'")
    lbl = r.get("label")
    if lbl not in VALID_LABELS:
        issues.append(f"label '{lbl}' not in {sorted(VALID_LABELS)}")
    for f in ("baseline", "result", "accuracy"):
        if f in r and not isinstance(r[f], dict):
            issues.append(f"'{f}' should be an object, got {type(r[f]).__name__}")
    s = r.get("speedup_graph", None)
    if not (s is None or isinstance(s, (int, float)) and not isinstance(s, bool)):
        issues.append(f"speedup_graph should be number or null, got {type(s).__name__}")
    for f in ("ts", "rig", "change", "kernel"):
        if f in r and not isinstance(r[f], str):
            issues.append(f"'{f}' should be a string")
    # cross-check: a positive/negative verdict with a headline ratio should have
    # the sign agree with the label (soft — flag, don't fail).
    if lbl == "positive" and rec.speedup_sign == "down":
        issues.append("label positive but speedup_graph < 1 (check sign)")
    if lbl == "negative" and rec.speedup_sign == "up":
        issues.append("label negative but speedup_graph > 1 (check sign)")
    return issues


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def corpus_files(path: str = DEFAULT_CORPUS) -> list:
    """Resolve `path` to a sorted list of JSONL files. `path` may be a directory
    (loads every *.jsonl inside — the default, all rig files), a glob pattern,
    or a single file."""
    if os.path.isdir(path):
        meta = os.path.basename(MODEL_META_FILE)
        return sorted(p for p in glob.glob(os.path.join(path, "*.jsonl"))
                      if os.path.basename(p) != meta)
    hits = sorted(glob.glob(path))
    return hits if hits else [path]


def load(path: str = DEFAULT_CORPUS, validate: bool = True) -> list:
    """Load a JSONL corpus into a list[Record], concatenating every rig file
    under `path` (see corpus_files). Raises on malformed JSON lines (that IS
    fatal); schema issues are attached to record.issues when validate=True.
    record.source records which rig file each row came from."""
    records = []
    for fp in corpus_files(path):
        src = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{fp}:{lineno}: bad JSON: {e}") from e
                rec = Record(idx=len(records), raw=obj, source=src)
                if validate:
                    rec.issues = validate_record(rec)
                records.append(rec)
    return records


def labeled(records: list) -> list:
    """Records whose label is a valid class — the only ones scorable by the LOO
    eval. Guards the loader against future non-conforming rows."""
    return [r for r in records if r.label in VALID_LABELS]


# ---------------------------------------------------------------------------
# Leave-one-out splitter (the eval protocol, fixed NOW)
# ---------------------------------------------------------------------------
def leave_one_out(records: list) -> Iterator[tuple]:
    """Yield (test_record, train_records) for every record. This is THE eval
    protocol for arm 2 across all model generations — LOO over labeled records,
    so retrieval / GBM / LoRA are all scored on identical folds. With ~45
    records LOO is the only defensible split (a held-out test set would be
    3-4 records wide)."""
    n = len(records)
    for i in range(n):
        test = records[i]
        train = [records[j] for j in range(n) if j != i]
        yield test, train


# ---------------------------------------------------------------------------
# Summary CLI
# ---------------------------------------------------------------------------
def summarize(records: list) -> dict:
    labels = Counter(r.label for r in records)
    comps = Counter(r.component for r in records)
    signs = Counter(r.speedup_sign for r in records)
    rigs = Counter(r.rig for r in records)
    per_rig = {rig: dict(Counter(r.label for r in records if r.rig == rig))
               for rig in rigs}
    with_ratio = sum(1 for r in records if r.signed_speedup is not None)
    n_issues = sum(1 for r in records if r.issues)
    return {
        "n": len(records),
        "labels": dict(labels),
        "components": dict(comps),
        "speedup_signs": dict(signs),
        "rigs": dict(rigs),
        "per_rig_labels": per_rig,
        "with_headline_ratio": with_ratio,
        "records_with_issues": n_issues,
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="validate + summarize the tune-data corpus")
    ap.add_argument("--file", default=DEFAULT_CORPUS, help="corpus JSONL path")
    args = ap.parse_args()

    records = load(args.file, validate=True)
    s = summarize(records)
    print(f"corpus: {args.file}")
    print(f"files: {[os.path.basename(f) for f in corpus_files(args.file)]}")
    print(f"records: {s['n']}")
    print(f"labels: {s['labels']}")
    print(f"per-rig: {s['rigs']}")
    for rig, lab in s["per_rig_labels"].items():
        print(f"    {rig}: {lab}")
    print(f"components: {s['components']}")
    print(f"speedup signs: {s['speedup_signs']}  (headline ratio present in {s['with_headline_ratio']}/{s['n']})")
    print(f"records with validation issues: {s['records_with_issues']}")
    for r in records:
        if r.issues:
            print(f"  [{r.idx}] {r.change[:60]!r}")
            for iss in r.issues:
                print(f"       - {iss}")
    # smoke-test the splitter and feature extractor so `dataset.py` is a
    # self-check: every fold must partition cleanly.
    folds = list(leave_one_out(records))
    assert len(folds) == s["n"], "LOO fold count mismatch"
    for test, train in folds:
        assert len(train) == s["n"] - 1
        assert test not in train
    _ = record_to_features(records[0])
    print(f"leave-one-out: {len(folds)} folds OK; feature keys = "
          f"{list(record_to_features(records[0]).keys())}")


if __name__ == "__main__":
    _main()
