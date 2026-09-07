#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("validate_re_hallu")

CATEGORIES = ["relation_hallucination", "incompleteness", "overgeneration", "context_induced"]

PROMPT_UNSAFE_KEYS = (
    "hallucination_provenance", "hallucination_category",
    "is_hallucination_benchmark", "source_benchmark_id", "metadata",
)

CATEGORY_GLOSS = {
    "relation_hallucination":
        "the arguments are right but at least one relation carries an incorrect label",
    "incompleteness":
        "every relation shown is correct, but one or more supported relations are missing",
    "overgeneration":
        "the correct relations are present, plus at least one unsupported extra relation",
    "context_induced":
        "an unsupported relation is present and the auxiliary context has been written to "
        "make it look plausible",
}

NEGATIVE_LABELS = {"DDI": {"NON"}}   # matches generate_re_hallu.py: NON is excluded from "gold"

SYSTEM_PROMPT = (
    "You are an independent biomedical annotator auditing a relation-extraction benchmark. "
    "You did not build this benchmark and have no stake in its results. Judge each change "
    "strictly against the evidence given. Return only valid JSON, no commentary."
)


# ============================== helpers ================================

def strip_reasoning(text: str) -> str:
    if "<think>" not in text:
        return text
    while "<think>" in text:
        s = text.find("<think>")
        e = text.find("</think>", s)
        if e == -1:
            return text[:s]
        text = text[:s] + text[e + len("</think>"):]
    return text.strip()


def extract_json_block(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = strip_reasoning(raw.strip()).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def strip_prompt_unsafe(row: Dict[str, Any]) -> Dict[str, Any]:
    clean = copy.deepcopy(row)
    for k in PROMPT_UNSAFE_KEYS:
        clean.pop(k, None)
    return clean


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer(v: Any, allowed: Sequence[str], default: str = "unparseable") -> str:
    s = str(v or "").strip().lower().replace(" ", "_")
    for a in allowed:
        if s == a.replace(" ", "_"):
            return a
    return default


# ========================= stratified sampling ==========================

def apportion_proportional(cell_sizes: Dict[Any, int], total: int,
                           min_per_cell: int = 0) -> Dict[Any, int]:
    """Largest-remainder (Hamilton) apportionment: each cell's sample count is
    proportional to its share of the full corpus, not equal across cells.

    Every cell gets floor(total * cell_size / corpus_size) first, then any
    leftover from rounding goes to the cells with the largest fractional
    remainder, one at a time, until `total` is reached exactly (subject to
    each cell's own availability).

    min_per_cell is applied as a floor AFTER proportional allocation, so a
    cell too small to earn any seats proportionally still gets checked at
    all. If the floors alone exceed `total` (e.g. many tiny cells with a
    high min_per_cell), the returned total legitimately exceeds `total` --
    this is intentional and logged by the caller, not silently absorbed.
    """
    cells = list(cell_sizes)
    corpus_total = sum(cell_sizes.values())
    if corpus_total == 0 or total <= 0:
        return {c: 0 for c in cells}

    raw = {c: total * cell_sizes[c] / corpus_total for c in cells}
    alloc = {c: min(int(raw[c]), cell_sizes[c]) for c in cells}
    remainder = total - sum(alloc.values())

    order = sorted(cells, key=lambda c: (raw[c] - int(raw[c])), reverse=True)
    i = 0
    guard = 0
    while remainder > 0 and any(alloc[c] < cell_sizes[c] for c in cells) and guard < 50 * max(1, len(cells)):
        c = order[i % len(order)]
        if alloc[c] < cell_sizes[c]:
            alloc[c] += 1
            remainder -= 1
        i += 1
        guard += 1

    if min_per_cell > 0:
        for c in cells:
            alloc[c] = max(alloc[c], min(min_per_cell, cell_sizes[c]))

    return alloc


def build_sample(hallu_rows: Sequence[Dict[str, Any]], sample_size: int,
                 per_cell_max: int, seed: int, strategy: str = "proportional",
                 min_per_cell: int = 0) -> List[Dict[str, Any]]:
    """Stratified sample over (dataset, category) cells.

    strategy='equal' (the original behaviour): every cell contributes before
    any cell contributes a second instance, so all 16 cells end up roughly
    equal-sized regardless of how large that cell actually is in the corpus.
    Good for guaranteeing small cells get checked as thoroughly as large
    ones; bad if you want the sample's composition to reflect what the
    deployed benchmark actually looks like.

    strategy='proportional' (new default): each cell's sample size mirrors
    its real share of the 17,653-instance corpus via apportion_proportional.
    A cell like DDI-overgeneration (3,020 real instances) gets far more
    review than BioRED-context_induced (485 real instances), the same way
    the benchmark itself is weighted -- so a pooled "X% of the sample is
    valid" number is actually representative of the benchmark as shipped,
    not an artifact of equal per-cell weighting. min_per_cell guards against
    a genuinely tiny cell getting zero coverage.

    IMPORTANT: switching strategy changes which instances get selected for a
    given seed. If you already have judgments computed against an existing
    --sample_state_file (equal-strategy), do not rebuild it with a
    different strategy and reuse the same path -- that silently invalidates
    every prior --agreement_between comparison. Build a new
    --sample_state_file for a new strategy.
    """
    rng = random.Random(seed)
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in hallu_rows:
        cat = r.get("hallucination_category")
        if cat not in CATEGORIES:
            continue
        by_cell[(r.get("dataset", "?"), cat)].append(r)
    for bucket in by_cell.values():
        rng.shuffle(bucket)
    cells = sorted(by_cell)

    if strategy == "proportional":
        cell_sizes = {c: len(by_cell[c]) for c in cells}
        target = apportion_proportional(cell_sizes, sample_size, min_per_cell)
        selected: List[Dict[str, Any]] = []
        for c in cells:
            n = target[c]
            if per_cell_max > 0:
                n = min(n, per_cell_max)
            selected.extend(by_cell[c][:n])
        rng.shuffle(selected)
        return selected

    # strategy == "equal": original round-robin behaviour, unchanged.
    selected = []
    depth = 0
    while len(selected) < sample_size:
        progressed = False
        for cell in cells:
            bucket = by_cell[cell]
            if depth < min(len(bucket), per_cell_max if per_cell_max > 0 else len(bucket)):
                selected.append(bucket[depth])
                progressed = True
                if len(selected) >= sample_size:
                    break
        if not progressed:
            break
        depth += 1
    return selected


def sample_composition_report(sample: Sequence[Dict[str, Any]],
                              corpus_rows: Sequence[Dict[str, Any]]) -> str:
    """Human-readable table comparing the sample's (dataset, category) mix
    against the full corpus's mix, so a proportional draw's fidelity (or an
    equal draw's deliberate departure from it) is visible before spending
    any validator time on it."""
    def counts(rows):
        c = Counter()
        for r in rows:
            cat = r.get("hallucination_category")
            if cat in CATEGORIES:
                c[(r.get("dataset", "?"), cat)] += 1
        return c

    corpus_c = counts(corpus_rows)
    sample_c = counts(sample)
    corpus_total = sum(corpus_c.values()) or 1
    sample_total = sum(sample_c.values()) or 1

    lines = [f"{'cell':32s} {'corpus n':>9s} {'corpus %':>9s} {'sample n':>9s} {'sample %':>9s}"]
    for cell in sorted(set(corpus_c) | set(sample_c)):
        cn, sn = corpus_c.get(cell, 0), sample_c.get(cell, 0)
        cp, sp = 100 * cn / corpus_total, 100 * sn / sample_total
        ds, cat = cell
        lines.append(f"{ds+':'+cat:32s} {cn:9d} {cp:8.1f}% {sn:9d} {sp:8.1f}%")
    lines.append(f"{'TOTAL':32s} {sum(corpus_c.values()):9d} {100.0:8.1f}% "
                 f"{sum(sample_c.values()):9d} {100.0:8.1f}%")
    return "\n".join(lines)


def category_block(rng: Optional[random.Random] = None) -> str:
    """List the four categories with their definitions, optionally shuffled
    per instance. The shuffle still matters even though intended_category is
    now shown directly: it stops the glossary's own listing order from
    nudging the reviewer's independent characterization toward whichever
    category happens to be printed first, the same first-listed-option
    effect measured earlier in this project's blind-mode runs."""
    cats = list(CATEGORIES)
    if rng is not None:
        rng.shuffle(cats)
    return "\n".join(f"   - {c}: {CATEGORY_GLOSS[c]}" for c in cats)


# ============================= diff computation ==========================

def positive_relations(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Relations a dataset actually treats as gold, excluding negative-label
    annotations (DDI's NON pairs). Matches generate_re_hallu.py's own
    gold_relations(include_negative=False), so "original" here means exactly
    what the construction pipeline used as gold when building the
    perturbation, not the raw row's full relation list."""
    neg = NEGATIVE_LABELS.get((row.get("dataset") or "").upper(), set())
    return [r for r in (row.get("relations") or []) if r.get("label") not in neg]


def _arg_key(rel: Dict[str, Any]) -> Tuple:
    return tuple(sorted((a.get("role", ""), str(a.get("entity_id", "")))
                        for a in (rel.get("arguments") or [])))


def diff_relations(original: Sequence[Dict[str, Any]],
                   modified: Sequence[Dict[str, Any]]) -> Dict[str, List]:
    """Compare two relation lists by argument identity, not just set
    membership, so a same-arguments-different-label edit is reported as a
    single "changed" item rather than an unrelated remove+add pair -- that
    distinction is exactly what a reviewer needs to judge the RIGHT thing
    (relation_hallucination looks like a "changed" entry; incompleteness
    only produces "removed" entries; overgeneration only "added")."""
    orig_by_args = {_arg_key(r): r.get("label", "") for r in original}
    mod_by_args = {_arg_key(r): r.get("label", "") for r in modified}
    added, removed, changed = [], [], []
    for k, lab in mod_by_args.items():
        if k not in orig_by_args:
            added.append((k, lab))
        elif orig_by_args[k] != lab:
            changed.append((k, orig_by_args[k], lab))
    for k, lab in orig_by_args.items():
        if k not in mod_by_args:
            removed.append((k, lab))
    return {"added": added, "removed": removed, "changed_label": changed}


def _entity_text(row: Dict[str, Any], entity_id: str) -> str:
    for e in row.get("entities") or []:
        if str(e.get("entity_id")) == str(entity_id):
            return e.get("text") or (e.get("surface_forms") or [str(entity_id)])[0]
    return str(entity_id)


def _readable_args(row: Dict[str, Any], arg_key: Tuple) -> str:
    return ", ".join(f"{role}={_entity_text(row, eid)}" for role, eid in arg_key)


def format_relation_list(row: Dict[str, Any], relations: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for r in relations:
        args = ", ".join(f"{a.get('role')}={_entity_text(row, a.get('entity_id'))}"
                         for a in (r.get("arguments") or []))
        parts.append(f"{r.get('label')}({args})")
    return "; ".join(parts) if parts else "(none)"


def format_change_summary(row: Dict[str, Any], diff: Dict[str, List]) -> str:
    """Human/model-readable description of exactly what changed, using real
    entity surface forms rather than raw IDs, so the reviewer never has to
    reconstruct the diff themselves -- it is already computed and stated."""
    lines = []
    for k, old_lab, new_lab in diff["changed_label"]:
        lines.append(f"LABEL CHANGED: ({_readable_args(row, k)}) was '{old_lab}', "
                     f"now labeled '{new_lab}'")
    for k, lab in diff["removed"]:
        lines.append(f"RELATION REMOVED: '{lab}'({_readable_args(row, k)}) is no longer present")
    for k, lab in diff["added"]:
        lines.append(f"RELATION ADDED: '{lab}'({_readable_args(row, k)}) is new, not in the "
                     f"original annotation")
    return "\n".join(lines) if lines else "(no relation-level difference detected)"


# ============================= prompting ================================

def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0 or len(text or "") <= max_chars:
        return (text or ""), False
    return text[:max_chars].rstrip() + " ...[truncated]", True


# ===================== comparative (non-blind) mode ========================
#
# Everything below shows the reviewer -- model or human -- the GOLD relation
# set alongside the MODIFIED (hallucinated) one, and asks them to judge the
# specific change, rather than judging the modified record in isolation and
# guessing what changed. This is a deliberately different question from blind
# mode: it removes the "can a reader notice one flipped label buried in a
# 10-relation BioRED abstract" detectability confound entirely (a real effect
# we measured in blind mode -- validity correlated with corruption_rate, i.e.
# how large a fraction of the relation set was touched) and asks the more
# targeted, and arguably more decision-relevant, question: given the exact
# change, is it a genuine, clinically meaningful misrepresentation of TEXT.
#
# is_hallucination stops being close to a tautology here. Every modified
# instance is *structurally* different from gold by construction (the
# pipeline enforces non-equivalence) -- so "is this different from gold" is
# not a real question once both are shown side by side. What is still a real
# question, and what the prompt asks: does the modification, given what TEXT
# actually states, constitute a genuine error (as opposed to a technically-
# different-from-gold reading that TEXT does not clearly rule out -- this
# happens for real: 18 ChemProt argument pairs in this corpus carry two
# different gold labels across separate mentions, so "different from this
# particular gold relation" does not always mean "wrong").
#
# category is no longer a blind guess -- with the diff visible it is closer
# to a confirmation/characterization question ("do you agree this specific
# change is best described as X"), which is still worth asking: it can
# surface cases where the pipeline's own taxonomy label does not match what
# an informed reviewer would call the same change.

COMPARATIVE_INSTRUCTION = """TASK: compare the GOLD and MODIFIED relation sets below for this RECORD
and judge the specific difference between them.

GOLD is the original, human-annotated relation set for TEXT. MODIFIED is a version of it
that a construction pipeline deliberately altered to build a hallucination-detection
benchmark, intending to introduce the kind of error named in intended_category (defined
below). CHANGE_SUMMARY states plainly, using real entity names, exactly which relations were
added, removed, or had their label changed -- read it, but also read GOLD and MODIFIED
yourself, since a mechanical diff will not catch every kind of change (e.g. a rewritten
CONTEXT passage carries no relation-level diff at all).

Reference -- what each intended_category is supposed to mean:
{category_block}

Answer four questions about the SPECIFIC CHANGE, not about GOLD or MODIFIED in isolation.
intended_category is shown to you; do not simply agree with it by default -- state your own
independent judgment even when it agrees, and especially when it does not (e.g. a change that
also drops other relations is really "incompleteness" too, even if intended_category says
only "relation_hallucination"):

1. is_hallucination: given what TEXT actually states, does the MODIFIED version misrepresent
   it in a way GOLD does not? This includes not just an added or mislabeled relation, but also
   a relation TEXT clearly supports that MODIFIED silently drops -- an incomplete relation set
   is a form of misrepresentation too (it implies TEXT supports less than it actually does),
   not merely a recall gap to overlook. Answer "false" if, despite differing from GOLD,
   MODIFIED is still a reading TEXT does not clearly rule out (this happens: the same argument
   pair can sometimes be described with more than one defensible label). Answer "true",
   "false", or "not sure".

2. category: which of the categories above best characterizes the change you are looking at,
   in your own judgment -- this may or may not match intended_category. If is_hallucination is
   "false" or "not sure", answer "none".

3. clinically_meaningful: would this specific change, if it appeared in a real extraction
   system's output, plausibly alter a clinical or biological conclusion (an incorrect drug
   interaction, a fabricated gene-disease link, a flipped direction of effect) -- as opposed
   to being technically different but practically inert? Answer "true", "false", "not sure",
   or "not_applicable" if is_hallucination is "false".

4. unambiguous: could a competent biomedical annotator, given the same GOLD, MODIFIED, and
   TEXT, reach a different conclusion than you did in good faith? "true" if clear-cut,
   "false" if a reasonable reviewer could disagree.

5. rationale: one or two sentences citing the specific relation(s) that changed and the part
   of TEXT that supports your judgment.

Return ONLY JSON:
{{"is_hallucination": "true"|"false"|"not sure",
  "category": "relation_hallucination"|"incompleteness"|"overgeneration"|"context_induced"|"none",
  "clinically_meaningful": "true"|"false"|"not sure"|"not_applicable",
  "unambiguous": "true"|"false",
  "rationale": "<short free text>"}}"""


def index_by_benchmark_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r.get("benchmark_id"): r for r in rows if r.get("benchmark_id")}


def render_comparison_record(hallu_row: Dict[str, Any], gold_row: Optional[Dict[str, Any]],
                             max_text_chars: int = 0, max_context_chars: int = 0,
                             max_entities: int = 0, max_relations: int = 0
                             ) -> Tuple[str, Dict[str, bool]]:
    """Shows TEXT, both versions of context_text (when the auxiliary context
    itself was rewritten -- context_induced), entities, GOLD and MODIFIED
    relations, the intended_category, and a readable CHANGE_SUMMARY (added /
    removed / relabeled, using real entity names, not raw IDs) computed by
    diff_relations. Full transparency, no blinding: nothing here is hidden
    from the reviewer, model or human.

    If gold_row is None (source row not found in --unified_jsonl -- this
    does happen; run()/run_build_sample_only/run_export_human_review all log
    how many sampled instances hit this), falls back to a self-contained
    modified-only view with a note explaining why gold and the diff are
    unavailable for this specific instance.
    """
    clean_mod = strip_prompt_unsafe(hallu_row)
    flags = {"text": False, "context": False, "entities": False, "relations": False}
    text, flags["text"] = truncate_text(clean_mod.get("text", ""), max_text_chars)
    ents = clean_mod.get("entities") or []
    flags["entities"] = bool(max_entities > 0 and len(ents) > max_entities)
    ents = ents[:max_entities] if max_entities > 0 else ents
    ent_view = [{"entity_id": str(e.get("entity_id")), "type": e.get("type"),
                "text": e.get("text") or (e.get("surface_forms") or [""])[0]} for e in ents]

    if gold_row is None:
        mod_ctx, flags["context"] = truncate_text(clean_mod.get("context_text", ""), max_context_chars)
        mod_rels = clean_mod.get("relations") or []
        flags["relations"] = bool(max_relations > 0 and len(mod_rels) > max_relations)
        mod_rels = mod_rels[:max_relations] if max_relations > 0 else mod_rels
        payload = {
            "dataset": clean_mod.get("dataset"), "annotation_level": clean_mod.get("annotation_level"),
            "task_type": clean_mod.get("task_type"), "text": text, "context_text": mod_ctx,
            "entities": ent_view, "intended_category": hallu_row.get("hallucination_category", ""),
            "modified_relations": format_relation_list(hallu_row, mod_rels),
            "_comparison_note": ("No matching source row was found in --unified_jsonl for this "
                                 "instance's source_benchmark_id, so GOLD and CHANGE_SUMMARY are "
                                 "unavailable -- judge MODIFIED against TEXT alone."),
        }
        if any(flags.values()):
            payload["_validator_note"] = ("One or more fields were truncated for length. Judge "
                                          "only what is visible.")
        return json.dumps(payload, ensure_ascii=False, indent=2), flags

    clean_gold = strip_prompt_unsafe(gold_row)
    gold_ctx = (clean_gold.get("context_text") or "").strip()
    mod_ctx = (clean_mod.get("context_text") or "").strip()
    context_block: Dict[str, Any]
    if gold_ctx != mod_ctx:
        gtxt, gflag = truncate_text(gold_ctx, max_context_chars)
        mtxt, mflag = truncate_text(mod_ctx, max_context_chars)
        flags["context"] = gflag or mflag
        context_block = {"original_context_text": gtxt, "modified_context_text": mtxt,
                         "_note": "The auxiliary context was rewritten; compare both versions."}
    else:
        ctxt, cflag = truncate_text(mod_ctx, max_context_chars)
        flags["context"] = cflag
        context_block = {"context_text": ctxt}

    orig_rels = positive_relations(gold_row)
    mod_rels = clean_mod.get("relations") or []
    if max_relations > 0:
        flags["relations"] = len(orig_rels) > max_relations or len(mod_rels) > max_relations
        orig_rels = orig_rels[:max_relations]
        mod_rels = mod_rels[:max_relations]

    diff = diff_relations(orig_rels, mod_rels)
    payload = {
        "dataset": clean_mod.get("dataset"), "annotation_level": clean_mod.get("annotation_level"),
        "task_type": clean_mod.get("task_type"), "text": text,
        **context_block,
        "entities": ent_view,
        "intended_category": hallu_row.get("hallucination_category", ""),
        "gold_relations": format_relation_list(hallu_row, orig_rels),
        "modified_relations": format_relation_list(hallu_row, mod_rels),
        "change_summary": format_change_summary(hallu_row, diff),
    }
    if any(flags.values()):
        payload["_validator_note"] = (
            "One or more fields were truncated for length before being shown to you. "
            "Judge only what is visible; do not assume anything about truncated content.")
    return json.dumps(payload, ensure_ascii=False, indent=2), flags


def build_comparison_prompt(hallu_row: Dict[str, Any], gold_row: Optional[Dict[str, Any]],
                            max_text_chars: int = 0, max_context_chars: int = 0,
                            max_entities: int = 0, max_relations: int = 0,
                            rng: Optional[random.Random] = None) -> Tuple[str, Dict[str, bool]]:
    instr = COMPARATIVE_INSTRUCTION.format(category_block=category_block(rng))
    rendered, flags = render_comparison_record(hallu_row, gold_row, max_text_chars,
                                               max_context_chars, max_entities, max_relations)
    return f"{instr}\n\nRECORD:\n{rendered}", flags


# ============================== backends ================================

def render_chat(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


class VLLMBackend:
    def __init__(self, model_path: str, system_prompt: str, max_new_tokens: int,
                max_model_len: int, tensor_parallel_size: int,
                gpu_memory_utilization: float, dtype: str, enforce_eager: bool, seed: int):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        LOGGER.info("Loading vLLM engine: %s (tp=%d, enforce_eager=%s)",
                    model_path, tensor_parallel_size, enforce_eager)
        self.llm = LLM(model=model_path, dtype=dtype, tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, trust_remote_code=True,
                       enforce_eager=enforce_eager, seed=seed)
        self.max_new_tokens = max_new_tokens

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        from vllm import SamplingParams
        rendered = [render_chat(self.tokenizer, self.system_prompt, p) for p in prompts]
        params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, n=1)
        outputs = self.llm.generate(rendered, params)
        results = [""] * len(rendered)
        for i, out in enumerate(outputs):
            try:
                pos = int(out.request_id)
            except (AttributeError, TypeError, ValueError):
                pos = i
            if not 0 <= pos < len(results):
                pos = i
            results[pos] = out.outputs[0].text.strip() if out.outputs else ""
        return results

    def close(self):
        pass


def build_openai_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


def call_with_retries(fn, max_retries: int, base_delay: float, sleep_fn=None):
    sleep = sleep_fn or (lambda _s: None)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - deliberately broad, network-facing call
            last_err = str(exc)
            if attempt < max_retries:
                sleep(base_delay * (2 ** attempt))
    return None, last_err


class APIBackend:
    """Same OpenAI-compatible design as run_e2e_llm.py / run_hallu_detection.py.
    Anthropic's native API is not covered (different request schema)."""

    def __init__(self, api_model: str, system_prompt: str, api_base_url: str, api_key_env: str,
                concurrency: int, max_retries: int, request_timeout: float, max_new_tokens: int):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'api' backend requires the openai package: pip install openai") from exc
        import os
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set.")
        self.client = OpenAI(api_key=key, base_url=api_base_url or None)
        self.model = api_model
        self.system_prompt = system_prompt
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = request_timeout
        self.max_new_tokens = max_new_tokens
        LOGGER.info("Using API validator: model=%s", self.model)

    def _one(self, prompt: str) -> str:
        messages = build_openai_messages(self.system_prompt, prompt)

        def call():
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0, timeout=self.timeout)
            return resp.choices[0].message.content or ""

        import time
        result, err = call_with_retries(call, self.max_retries, 1.0, time.sleep)
        if err is not None:
            LOGGER.warning("API call failed after retries: %s", err)
            return ""
        return result

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, prompts))

    def close(self):
        pass


class EchoBackend:
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ['{"is_hallucination": "not sure", "category": "none", '
                                       '"clinically_meaningful": "not_applicable", '
                                       '"unambiguous": "false", "rationale": ""}']

    def generate_batch(self, prompts):
        return [self.responses[i % len(self.responses)] for i in range(len(prompts))]

    def close(self):
        pass


# ========================= human review export/import =====================

HUMAN_REVIEW_COLUMNS = [
    "review_id", "benchmark_id", "dataset", "intended_category", "text",
    "original_context_text", "modified_context_text", "entities",
    "gold_relations", "modified_relations", "change_summary",
    "is_hallucination", "category", "clinically_meaningful", "unambiguous", "rationale",
]

ANSWER_KEY_FILENAME = "_answer_key_DO_NOT_SHARE.json"


def _readable_entities(entities: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for e in entities or []:
        text = e.get("text") or (e.get("surface_forms") or [""])[0]
        parts.append(f"{e.get('entity_id')}:{e.get('type')}:{text}")
    return "; ".join(parts)


def export_for_human_review(sample: Sequence[Dict[str, Any]], output_dir: str, seed: int,
                            gold_index: Optional[Dict[str, Dict[str, Any]]] = None
                            ) -> Tuple[str, str]:
    """Write ONE combined, self-contained CSV covering the whole sample --
    every row carries its own benchmark_id and intended_category as ordinary
    columns, so the file needs no companion answer key to be usable. Built
    for WebForm-style pipelines: hand this single file to a form generator,
    get a single file of responses back, and load_human_review() reads it
    with nothing else required. dataset and intended_category are still
    present as regular columns if you want to filter/sort within the sheet
    or the form tool itself.

    Shows GOLD and MODIFIED relations, both versions of context_text when it
    was rewritten, intended_category, and a readable change_summary --
    identical content and formatting to what the LLM validators see via
    build_comparison_prompt, so human and model judgments are comparing
    against the same view of the evidence, which is what makes merging their
    results into one final assessment (see compute_verdict) meaningful.
    Requires gold_index (built from --unified_jsonl); a source row missing
    from it is exported with the modified-only view and a note explaining
    why, same fallback as the LLM prompt path.

    An answer key is still written alongside as a redundant backup (in case
    a WebForm tool strips or reorders columns on export) but is no longer
    required for anything to work -- load_human_review() reads benchmark_id
    and intended_category straight from the CSV itself.
    """
    rng = random.Random(seed)
    rows_all = [r for r in sample if r.get("hallucination_category") in CATEGORIES]
    rng.shuffle(rows_all)

    out_dir = Path(output_dir) / "human_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "human_review.csv"

    answer_key: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HUMAN_REVIEW_COLUMNS)
        for counter, r in enumerate(rows_all, start=1):
            rid = f"R{counter:05d}"
            ds = r.get("dataset", "?")
            intended = r.get("hallucination_category", "")
            bid = r.get("benchmark_id", "")
            gold_row = (gold_index or {}).get(r.get("source_benchmark_id"))
            clean_mod = strip_prompt_unsafe(r)

            if gold_row is not None:
                clean_gold = strip_prompt_unsafe(gold_row)
                gold_ctx = (clean_gold.get("context_text") or "").strip()
                mod_ctx = (clean_mod.get("context_text") or "").strip()
                orig_rels = positive_relations(gold_row)
                mod_rels = clean_mod.get("relations") or []
                diff = diff_relations(orig_rels, mod_rels)
                writer.writerow([
                    rid, bid, ds, intended, clean_mod.get("text", ""),
                    gold_ctx, mod_ctx,
                    _readable_entities(clean_mod.get("entities")
                                      or clean_gold.get("entities") or []),
                    format_relation_list(r, orig_rels), format_relation_list(r, mod_rels),
                    format_change_summary(r, diff),
                    "", "", "", "", "",
                ])
            else:
                writer.writerow([
                    rid, bid, ds, intended, clean_mod.get("text", ""),
                    "(source row not found in --unified_jsonl)", clean_mod.get("context_text", ""),
                    _readable_entities(clean_mod.get("entities") or []),
                    "(unavailable)", format_relation_list(r, clean_mod.get("relations") or []),
                    "(unavailable -- no gold row to diff against)",
                    "", "", "", "", "",
                ])
            answer_key[rid] = {"benchmark_id": bid, "dataset": ds, "intended_category": intended}

    key_path = out_dir / ANSWER_KEY_FILENAME
    with open(key_path, "w", encoding="utf-8") as fh:
        json.dump(answer_key, fh, ensure_ascii=False, indent=2)
    return str(csv_path), str(key_path)


def load_human_review(csv_paths: Sequence[str],
                      answer_key_path: Optional[str] = None) -> Tuple[List["Judgment"], int]:
    """Read completed human-review CSV(s) back into Judgment objects.

    Self-contained by default: benchmark_id, dataset, and intended_category
    are read straight from the CSV's own columns, so a single WebForm export
    is enough on its own. answer_key_path is optional and used only as a
    fallback for any row missing one of those columns (e.g. a very old
    export, or a form tool that stripped a column on the way out) --
    when given, it fills gaps rather than being required for every row.

    Returns (judgments, n_incomplete). A row whose is_hallucination column is
    still blank is treated as not-yet-reviewed and excluded from the
    returned judgments (not counted as "unparseable", which would wrongly
    imply the annotator gave an unreadable answer rather than no answer at
    all) -- n_incomplete tells you how many rows still need attention.
    """
    answer_key: Dict[str, Dict[str, str]] = {}
    if answer_key_path:
        with open(answer_key_path, "r", encoding="utf-8") as fh:
            answer_key = json.load(fh)

    judgments: List[Judgment] = []
    n_incomplete = 0
    for path in csv_paths:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                rid = (row.get("review_id") or "").strip()
                bid = (row.get("benchmark_id") or "").strip()
                ds = (row.get("dataset") or "").strip()
                intended = (row.get("intended_category") or "").strip()

                if not (bid and ds and intended):
                    meta = answer_key.get(rid)
                    if meta is None:
                        LOGGER.warning("Row %s in %s is missing benchmark_id/dataset/"
                                      "intended_category and has no answer-key fallback "
                                      "for review_id %s; skipping.", rid, path, rid)
                        continue
                    bid = bid or meta.get("benchmark_id", "")
                    ds = ds or meta.get("dataset", "")
                    intended = intended or meta.get("intended_category", "")

                if not (row.get("is_hallucination") or "").strip():
                    n_incomplete += 1
                    continue
                is_h = normalize_answer(row.get("is_hallucination"), ["true", "false", "not_sure"])
                stated_cat = normalize_answer(row.get("category"), CATEGORIES + ["none"])
                clin = normalize_answer(row.get("clinically_meaningful"),
                                        ["true", "false", "not_sure", "not_applicable"])
                unamb = normalize_answer(row.get("unambiguous"), ["true", "false"])
                rationale = (row.get("rationale") or "")[:500]
                match = (stated_cat == intended) if is_h == "true" else None
                judgments.append(Judgment(bid, ds, intended, is_h, stated_cat, clin, unamb,
                                          rationale, match))
    return judgments, n_incomplete


# ============================= judgment parsing ==========================

@dataclass
class Judgment:
    benchmark_id: str
    dataset: str
    intended_category: str
    is_hallucination: str        # true|false|not_sure|unparseable
    stated_category: str          # one of CATEGORIES | none | unparseable
    # In --mode blind: a genuine blind guess, made without seeing the
    # intended category or the gold relation set. In --mode comparative
    # (the default): a characterization made WITH gold and modified both
    # visible -- closer to "do you agree this change should be called X"
    # than to a guess. Which mode produced a given judgments_*.jsonl is
    # always recorded in its config block; check that before comparing
    # stated_category numbers across two runs, since the two modes are
    # measuring different things and are not directly comparable.
    clinically_meaningful: str
    unambiguous: str
    rationale: str
    category_match: Optional[bool]  # None if is_hallucination isn't "true"


def parse_judgment(row: Dict[str, Any], raw: str) -> Judgment:
    obj = extract_json_block(raw)
    intended = row.get("hallucination_category", "")
    if obj is None:
        return Judgment(row.get("benchmark_id", ""), row.get("dataset", ""), intended,
                        "unparseable", "unparseable", "unparseable", "unparseable", "", None)
    is_hallu = normalize_answer(obj.get("is_hallucination"), ["true", "false", "not_sure"])
    stated_cat = normalize_answer(obj.get("category"), CATEGORIES + ["none"])
    clin = normalize_answer(obj.get("clinically_meaningful"),
                            ["true", "false", "not_sure", "not_applicable"])
    unamb = normalize_answer(obj.get("unambiguous"), ["true", "false"])
    rationale = str(obj.get("rationale", ""))[:500]

    # Enforce server-side the same consistency rule the human review tool
    # already applies client-side: category is only meaningful when
    # is_hallucination is "true". Without this, a model can emit
    # is_hallucination="false" alongside category="incompleteness" in the
    # same response -- exactly the pattern that made incompleteness's
    # per-category recall look near-100% (both validators correctly named
    # the omission) while its semantic_validity_rate was near 0 (both then
    # separately declined to call it a hallucination). Forcing category to
    # "none" whenever is_hallucination isn't "true" stops that internal
    # contradiction from silently inflating category-level metrics for a
    # category the model's own is_hallucination answer is rejecting.
    if is_hallu != "true":
        stated_cat = "none"
    if is_hallu == "false":
        clin = "not_applicable"

    match = (stated_cat == intended) if is_hallu == "true" else None
    return Judgment(row.get("benchmark_id", ""), row.get("dataset", ""), intended,
                    is_hallu, stated_cat, clin, unamb, rationale, match)


# ================================ metrics =================================

def summarize(judgments: Sequence[Judgment]) -> Dict[str, Any]:
    n = len(judgments)
    is_hallu_counts = Counter(j.is_hallucination for j in judgments)
    matched = [j for j in judgments if j.category_match is not None]
    match_rate = (sum(j.category_match for j in matched) / len(matched)) if matched else None

    per_cell: Dict[str, Dict[str, Any]] = {}
    by_cell: Dict[Tuple[str, str], List[Judgment]] = defaultdict(list)
    for j in judgments:
        by_cell[(j.dataset, j.intended_category)].append(j)
    for (ds, cat), js in sorted(by_cell.items()):
        m = [j for j in js if j.category_match is not None]
        per_cell[f"{ds}:{cat}"] = {
            "n": len(js),
            "semantic_validity_rate": round(
                sum(j.is_hallucination == "true" for j in js) / len(js), 4) if js else 0.0,
            "stated_category_agreement": round(sum(j.category_match for j in m) / len(m), 4)
                                        if m else None,
            "clinically_meaningful_rate": round(
                sum(j.clinically_meaningful == "true" for j in js
                    if j.clinically_meaningful != "not_applicable")
                / max(1, sum(j.clinically_meaningful != "not_applicable" for j in js)), 4),
            "unambiguous_rate": round(sum(j.unambiguous == "true" for j in js) / len(js), 4),
            "not_sure_rate": round(sum(j.is_hallucination == "not_sure" for j in js) / len(js), 4),
        }

    confusion: Dict[str, Counter] = {c: Counter() for c in CATEGORIES}
    for j in judgments:
        if j.intended_category in confusion:
            confusion[j.intended_category][j.stated_category] += 1

    # Precision/recall per stated category label, not just raw accuracy.
    # Accuracy alone is misleading whenever the validator has a positional
    # or default-answer bias: if it guesses one label on 90% of instances
    # regardless of evidence, recall for that label looks inflated and
    # accuracy for every other label looks collapsed, even though the
    # *quality* of a correct guess (precision) may be fine everywhere.
    # This is exactly what happened here before category order was
    # randomized -- see category_block().
    per_category_stats: Dict[str, Dict[str, Any]] = {}
    for c in CATEGORIES:
        tp = confusion[c][c]
        n_true = sum(confusion[c].values())
        n_pred = sum(confusion[other].get(c, 0) for other in CATEGORIES)
        recall = round(tp / n_true, 4) if n_true else None
        precision = round(tp / n_pred, 4) if n_pred else None
        f1 = (round(2 * precision * recall / (precision + recall), 4)
              if precision and recall and (precision + recall) > 0 else None)
        per_category_stats[c] = {
            "n_true": n_true, "n_guessed": n_pred, "true_positives": tp,
            "precision": precision, "recall": recall, "f1": f1,
        }
    guess_distribution = {c: sum(confusion[t].get(c, 0) for t in CATEGORIES) for c in CATEGORIES}
    max_guess_share = round(max(guess_distribution.values()) / n, 4) if n else 0.0

    return {
        "n_instances": n,
        "is_hallucination_distribution": dict(is_hallu_counts),
        "semantic_validity_rate": round(is_hallu_counts.get("true", 0) / n, 4) if n else 0.0,
        "stated_category_agreement_overall": round(match_rate, 4) if match_rate is not None else None,
        "per_cell": per_cell,
        "stated_category_confusion": {k: dict(v) for k, v in confusion.items()},
        "per_category_precision_recall": per_category_stats,
        "stated_category_distribution": guess_distribution,
        "max_single_label_guess_share": max_guess_share,
        "_bias_warning": (
            f"One stated category label was guessed on {max_guess_share*100:.1f}% of all "
            f"{n} instances. If this is well above 25% (chance for 4 balanced categories), "
            "check per_category_precision_recall rather than trusting raw accuracy/recall: "
            "a dominant default-answer bias inflates that label's recall and deflates every "
            "other label's recall without reflecting true category-level quality."
        ) if max_guess_share > 0.40 else None,
        "unparseable_rate": round(sum(j.is_hallucination == "unparseable"
                                      for j in judgments) / n, 4) if n else 0.0,
    }


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Standard two-rater, nominal-category Cohen's kappa."""
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return 0.0
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    cats = sorted(set(labels_a) | set(labels_b))
    pa = {c: sum(a == c for a in labels_a) / n for c in cats}
    pb = {c: sum(b == c for b in labels_b) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def compute_agreement(judgments_a: Sequence[Judgment], judgments_b: Sequence[Judgment]) -> Dict[str, Any]:
    """Inter-validator agreement, matched by benchmark_id. Two validator
    runs over the same sample (not necessarily the same order)."""
    by_id_b = {j.benchmark_id: j for j in judgments_b}
    paired = [(a, by_id_b[a.benchmark_id]) for a in judgments_a if a.benchmark_id in by_id_b]
    if not paired:
        return {"n_paired": 0, "error": "no overlapping benchmark_id between the two files"}

    hallu_a = [a.is_hallucination for a, _ in paired]
    hallu_b = [b.is_hallucination for _, b in paired]
    cat_a = [a.stated_category for a, _ in paired]
    cat_b = [b.stated_category for _, b in paired]

    return {
        "n_paired": len(paired),
        "is_hallucination_percent_agreement": round(
            sum(a == b for a, b in zip(hallu_a, hallu_b)) / len(paired), 4),
        "is_hallucination_kappa": cohens_kappa(hallu_a, hallu_b),
        "stated_category_percent_agreement": round(
            sum(a == b for a, b in zip(cat_a, cat_b)) / len(paired), 4),
        "stated_category_kappa": cohens_kappa(cat_a, cat_b),
    }


# ============================= accept/regenerate verdict ===================

DEFAULT_VALIDITY_THRESHOLDS = (0.70, 0.40)   # >= first: accept; >= second: adjudicate; else: regenerate
DEFAULT_KAPPA_THRESHOLDS = (0.40, 0.20)      # Landis & Koch: moderate+ / fair / slight-poor


def _majority_true(votes: Sequence[str]) -> bool:
    return sum(v == "true" for v in votes) * 2 > len(votes)


def compute_verdict(judgments_by_validator: Dict[str, Sequence[Judgment]],
                    validity_thresholds: Tuple[float, float] = DEFAULT_VALIDITY_THRESHOLDS,
                    kappa_thresholds: Tuple[float, float] = DEFAULT_KAPPA_THRESHOLDS
                    ) -> Dict[str, Any]:
    """Per-(dataset, category)-cell accept / adjudicate / regenerate verdict
    from >=2 named validators (any mix of model and human judgment sets).

    Never decided from a single pooled number: a benchmark-wide average can
    hide a cell that is 95% valid sitting next to one that is 10% valid.
    Every cell gets its own verdict from two independent axes:

      consensus_validity  fraction of instances where a MAJORITY of
                           validators independently said "true" -- for two
                           validators this means both agree; for three,
                           at least two of three. Protects against one
                           lenient and one strict validator averaging out
                           to something meaningless (this is exactly what
                           happened comparing early Mistral-Nemo (~31-38%)
                           against Phi-4 (~97%) runs on the same instances).

      reliability_kappa   mean pairwise Cohen's kappa on is_hallucination
                          across all validator pairs, restricted to that
                          cell. Low kappa with high validity means
                          validators mostly say "yes" but for different
                          reasons/on different instances -- worth knowing
                          even when the raw rate looks fine.

    verdict per cell:
      accept       consensus_validity >= thresholds[0] AND kappa >= thresholds[0]
      adjudicate   consensus_validity >= thresholds[1] AND kappa >= thresholds[1]
                   (send to human review rather than decide by model alone)
      regenerate   below both -- but see the note on deterministic categories
                   in the module docstring before assuming this means
                   "rerun a generator": relation_hallucination, incompleteness,
                   and overgeneration are schema-exact by construction, so a
                   red verdict there more often means the perturbation was
                   too subtle to be visible (check corruption_rate) or the
                   validator's context window truncated the evidence, not
                   that construction itself is broken. Only context_induced
                   has an actual generation step to rerun.
    """
    names = sorted(judgments_by_validator)
    if len(names) < 2:
        raise ValueError("compute_verdict needs at least 2 named validators.")

    # index each validator's judgments by benchmark_id
    by_name_id: Dict[str, Dict[str, Judgment]] = {
        n: {j.benchmark_id: j for j in judgments_by_validator[n]} for n in names
    }
    common_ids = set.intersection(*[set(d) for d in by_name_id.values()])

    by_cell: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for bid in common_ids:
        j0 = by_name_id[names[0]][bid]
        by_cell[(j0.dataset, j0.intended_category)].append(bid)

    cells_out: Dict[str, Any] = {}
    counts = Counter()
    for cell, ids in sorted(by_cell.items()):
        ds, cat = cell
        n = len(ids)
        votes_per_id = {bid: [by_name_id[nm][bid].is_hallucination for nm in names] for bid in ids}
        consensus = sum(_majority_true(votes_per_id[bid]) for bid in ids) / n if n else 0.0

        kappas = []
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                a = [by_name_id[names[i]][bid].is_hallucination for bid in ids]
                b = [by_name_id[names[k]][bid].is_hallucination for bid in ids]
                if a and b:
                    kappas.append(cohens_kappa(a, b))
        mean_kappa = round(sum(kappas) / len(kappas), 4) if kappas else None

        if (consensus >= validity_thresholds[0]
                and (mean_kappa is None or mean_kappa >= kappa_thresholds[0])):
            verdict = "accept"
        elif (consensus >= validity_thresholds[1]
                and (mean_kappa is None or mean_kappa >= kappa_thresholds[1])):
            verdict = "adjudicate"
        else:
            verdict = "regenerate"
        counts[verdict] += 1

        cells_out[f"{ds}:{cat}"] = {
            "n": n, "consensus_validity_rate": round(consensus, 4),
            "mean_pairwise_kappa": mean_kappa, "verdict": verdict,
            "construction": ("deterministic" if cat != "context_induced" else "llm"),
        }

    return {
        "validators": names,
        "n_instances_common_to_all_validators": len(common_ids),
        "thresholds": {"validity": list(validity_thresholds), "kappa": list(kappa_thresholds)},
        "verdict_counts": dict(counts),
        "cells": cells_out,
    }


# ================================ driver ==================================

def run_build_sample_only(args) -> int:
    """Build and save the stratified sample with no model, no backend, no
    GPU. Run this ONCE, sequentially, before launching any validators in
    parallel -- both processes then only ever READ the finished file,
    which is race-free. Writing it from two concurrently-launched
    validator processes is NOT safe: each does a plain
    'exists? no -> build -> open(path, "w")' check-then-write with no
    lock, and a real vLLM model load (tens of seconds) sits in the gap
    between the check and the write, which is more than enough window for
    both processes to see "doesn't exist yet" and then race to create it.
    """
    if not args.sample_state_file:
        print("--build_sample_only requires --sample_state_file.", file=sys.stderr)
        return 2
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    hallu_rows = load_jsonl(args.hallu_jsonl)
    LOGGER.info("Loaded %d hallucination rows.", len(hallu_rows))
    sample = build_sample(hallu_rows, args.sample_size, args.per_cell_max, args.seed,
                          strategy=args.sample_strategy, min_per_cell=args.min_per_cell)
    LOGGER.info("Built stratified sample: %d instances (strategy=%s).",
               len(sample), args.sample_strategy)
    LOGGER.info("\n%s", sample_composition_report(sample, hallu_rows))

    Path(args.sample_state_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.sample_state_file, "w", encoding="utf-8") as fh:
        json.dump([r.get("benchmark_id") for r in sample], fh, indent=2)
    LOGGER.info("Saved to %s. Safe to launch any number of validators in "
               "parallel now -- they will only read this file.", args.sample_state_file)

    if args.export_human_review:
        if not args.unified_jsonl:
            print("--export_human_review requires --unified_jsonl (the source of the gold "
                 "relation sets shown alongside the modified ones).", file=sys.stderr)
            return 2
        unified_rows = load_jsonl(args.unified_jsonl)
        gold_index = index_by_benchmark_id(unified_rows)
        missing = sum(1 for r in sample if r.get("source_benchmark_id") not in gold_index)
        if missing:
            LOGGER.warning("%d/%d sampled instances have no matching source row in "
                          "--unified_jsonl; those will export with the modified-only "
                          "view.", missing, len(sample))
        csv_path, key_path = export_for_human_review(sample, args.output_dir, args.seed,
                                                      gold_index=gold_index)
        LOGGER.info("Wrote one combined human-review CSV (%d rows): %s", len(sample), csv_path)
        LOGGER.info("Answer key (backup only, not required): %s", key_path)
    return 0


def run_export_human_review(args) -> int:
    """Build (or reuse) the sample and export it as per-cell CSVs for human
    annotators, without needing --build_sample_only first."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    hallu_rows = load_jsonl(args.hallu_jsonl)
    LOGGER.info("Loaded %d hallucination rows.", len(hallu_rows))

    if args.sample_state_file and Path(args.sample_state_file).exists():
        ids = set(json.load(open(args.sample_state_file, encoding="utf-8")))
        sample = [r for r in hallu_rows if r.get("benchmark_id") in ids]
        LOGGER.info("Reusing existing sample from %s: %d instances.",
                   args.sample_state_file, len(sample))
    else:
        sample = build_sample(hallu_rows, args.sample_size, args.per_cell_max, args.seed,
                              strategy=args.sample_strategy, min_per_cell=args.min_per_cell)
        LOGGER.info("Built new sample: %d instances (strategy=%s).",
                   len(sample), args.sample_strategy)
        if args.sample_state_file:
            Path(args.sample_state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.sample_state_file, "w", encoding="utf-8") as fh:
                json.dump([r.get("benchmark_id") for r in sample], fh, indent=2)

    LOGGER.info("\n%s", sample_composition_report(sample, hallu_rows))

    if not args.unified_jsonl:
        print("--unified_jsonl is required.", file=sys.stderr)
        return 2
    unified_rows = load_jsonl(args.unified_jsonl)
    gold_index = index_by_benchmark_id(unified_rows)
    missing = sum(1 for r in sample if r.get("source_benchmark_id") not in gold_index)
    if missing:
        LOGGER.warning("%d/%d sampled instances have no matching source row in "
                      "--unified_jsonl; those will export with the modified-only view.",
                      missing, len(sample))

    csv_path, key_path = export_for_human_review(sample, args.output_dir, args.seed,
                                                 gold_index=gold_index)
    LOGGER.info("Wrote one combined human-review CSV (%d rows): %s", len(sample), csv_path)
    LOGGER.info("Answer key (backup only, not required): %s", key_path)
    return 0


def run_import_human_review(args) -> int:
    """Parse completed human-review CSVs into the same judgments_*.jsonl /
    summary_*.json format the model validators produce, so --agreement_between
    and --verdict work identically on human and model results."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    judgments, n_incomplete = load_human_review(args.import_human_review, args.answer_key)
    LOGGER.info("Loaded %d completed reviews (%d rows still blank / not yet reviewed).",
               len(judgments), n_incomplete)
    if not judgments:
        LOGGER.error("No completed reviews found -- nothing to write.")
        return 1

    summary = summarize(judgments)
    summary["config"] = {"validator_name": args.human_validator_name, "backend": "human",
                         "n_incomplete_at_import_time": n_incomplete,
                         "source_csvs": list(args.import_human_review)}
    tag = args.human_validator_name
    with open(out_dir / f"judgments_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for j in judgments:
            fh.write(json.dumps({
                "benchmark_id": j.benchmark_id, "dataset": j.dataset,
                "intended_category": j.intended_category,
                "is_hallucination": j.is_hallucination, "stated_category": j.stated_category,
                "category_match": j.category_match,
                "clinically_meaningful": j.clinically_meaningful,
                "unambiguous": j.unambiguous, "rationale": j.rationale,
                "raw_response": None,
            }, ensure_ascii=False) + "\n")
    with open(out_dir / f"summary_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote judgments_%s.jsonl and summary_%s.json to %s", tag, tag, out_dir)
    LOGGER.info("Semantic validity rate: %.3f", summary["semantic_validity_rate"])
    return 0


def run_verdict(paths: Sequence[str], validity_thresholds: Tuple[float, float],
                kappa_thresholds: Tuple[float, float]) -> int:
    """Load 2+ judgments_*.jsonl files (any mix of model and human), compute
    the per-cell accept/adjudicate/regenerate verdict, and print it."""
    def load(path: str) -> List[Judgment]:
        out = []
        for row in load_jsonl(path):
            out.append(Judgment(row["benchmark_id"], row["dataset"], row["intended_category"],
                                row["is_hallucination"], row["stated_category"],
                                row.get("clinically_meaningful", ""), row.get("unambiguous", ""),
                                row.get("rationale", ""), row.get("category_match")))
        return out

    def name_from_path(path: str) -> str:
        stem = Path(path).stem
        return stem[len("judgments_"):] if stem.startswith("judgments_") else stem

    by_validator = {name_from_path(p): load(p) for p in paths}
    verdict = compute_verdict(by_validator, validity_thresholds, kappa_thresholds)
    print(json.dumps(verdict, indent=2))
    return 0


def run(args) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_dir / "validation.log",
                                                      encoding="utf-8")],
                        force=True)

    hallu_rows = load_jsonl(args.hallu_jsonl)
    LOGGER.info("Loaded %d hallucination rows.", len(hallu_rows))

    if args.sample_state_file and Path(args.sample_state_file).exists():
        ids = set(json.load(open(args.sample_state_file, encoding="utf-8")))
        sample = [r for r in hallu_rows if r.get("benchmark_id") in ids]
        LOGGER.info("Reusing existing sample from %s: %d instances "
                   "(so a second validator judges the identical set).",
                   args.sample_state_file, len(sample))
    elif args.sample_state_file and args.require_prebuilt_sample:
        LOGGER.error("--sample_state_file %s does not exist and "
                    "--require_prebuilt_sample was set. Run with "
                    "--build_sample_only first (once, sequentially), then "
                    "launch validators in parallel -- see the module "
                    "docstring for why building it from a validator process "
                    "itself is not safe under concurrent launch.",
                    args.sample_state_file)
        return 1
    else:
        sample = build_sample(hallu_rows, args.sample_size, args.per_cell_max, args.seed,
                              strategy=args.sample_strategy, min_per_cell=args.min_per_cell)
        LOGGER.info("Built stratified sample: %d instances (strategy=%s).",
                   len(sample), args.sample_strategy)
        if args.sample_state_file:
            with open(args.sample_state_file, "w", encoding="utf-8") as fh:
                json.dump([r.get("benchmark_id") for r in sample], fh, indent=2)
            LOGGER.info("Sample IDs saved to %s -- pass the same path for a second "
                       "validator to judge the identical instances.", args.sample_state_file)

    LOGGER.info("Sample composition: %s",
               dict(Counter((r.get("dataset"), r.get("hallucination_category")) for r in sample)))

    if not args.unified_jsonl:
        LOGGER.error("--unified_jsonl is required (the source of the gold relation sets "
                    "shown alongside the modified ones).")
        return 2
    unified_rows = load_jsonl(args.unified_jsonl)
    gold_index = index_by_benchmark_id(unified_rows)
    missing = sum(1 for r in sample if r.get("source_benchmark_id") not in gold_index)
    if missing:
        LOGGER.warning("%d/%d sampled instances have no matching source row in "
                      "--unified_jsonl; those will be judged with the modified-only "
                      "view (fallback), which changes what the validator is judging "
                      "for exactly those instances -- worth checking why they're "
                      "missing before trusting their results.", missing, len(sample))

    system_prompt = SYSTEM_PROMPT
    if args.backend == "vllm":
        backend = VLLMBackend(args.model_path, system_prompt, args.max_new_tokens,
                              args.max_model_len, args.tensor_parallel_size,
                              args.gpu_memory_utilization, args.dtype, args.enforce_eager, args.seed)
    elif args.backend == "api":
        backend = APIBackend(args.api_model or args.model_path, system_prompt, args.api_base_url,
                             args.api_key_env, args.api_concurrency, args.api_max_retries,
                             args.api_request_timeout, args.max_new_tokens)
    else:
        backend = EchoBackend()

    prompts_flags = [
        build_comparison_prompt(r, gold_index.get(r.get("source_benchmark_id")),
                                args.max_text_chars, args.max_context_chars,
                                args.max_entities, args.max_relations,
                                rng=random.Random(f"{args.seed}:{r.get('benchmark_id', i)}"))
        for i, r in enumerate(sample)
        ]
    prompts = [p for p, _ in prompts_flags]
    trunc_counts = Counter()
    for _, flags in prompts_flags:
        for field, was_truncated in flags.items():
            if was_truncated:
                trunc_counts[field] += 1
    if sum(trunc_counts.values()):
        LOGGER.warning("Truncation applied to keep prompts under the context window: %s "
                       "(out of %d sampled instances). Increase --max_text_chars / "
                       "--max_entities / --max_relations, or --max_model_len, if you "
                       "want this at 0.", dict(trunc_counts), len(sample))
    else:
        LOGGER.info("No truncation needed for any of the %d sampled prompts at current "
                   "--max_text_chars/--max_entities/--max_relations settings.", len(sample))
    raws: List[str] = []
    try:
        for s in range(0, len(prompts), args.batch_size):
            raws.extend(backend.generate_batch(prompts[s:s + args.batch_size]))
            LOGGER.info("  %d/%d judged", min(s + args.batch_size, len(prompts)), len(prompts))
    finally:
        backend.close()

    judgments = [parse_judgment(r, raw) for r, raw in zip(sample, raws)]
    summary = summarize(judgments)
    summary["config"] = {
        "validator_name": args.validator_name, "backend": args.backend,
        "model": args.api_model or args.model_path, "sample_size_requested": args.sample_size,
        "sample_strategy": args.sample_strategy,
        "per_cell_max": args.per_cell_max, "seed": args.seed,
        "_note": ("Gold and modified relations were both shown to the validator alongside "
                 "intended_category; stated_category is an independent characterization made "
                 "with full context, not a blind guess."),
    }

    tag = args.validator_name or "validator"
    with open(out_dir / f"judgments_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r, j, raw in zip(sample, judgments, raws):
            fh.write(json.dumps({
                "benchmark_id": j.benchmark_id, "dataset": j.dataset,
                "intended_category": j.intended_category,
                "is_hallucination": j.is_hallucination, "stated_category": j.stated_category,
                "category_match": j.category_match,
                "clinically_meaningful": j.clinically_meaningful,
                "unambiguous": j.unambiguous, "rationale": j.rationale,
                "raw_response": raw if args.save_raw else None,
            }, ensure_ascii=False) + "\n")
    with open(out_dir / f"summary_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    LOGGER.info("=" * 66)
    LOGGER.info("Validator: %s", tag)
    LOGGER.info("Semantic validity rate (is_hallucination=true): %.3f",
               summary["semantic_validity_rate"])
    LOGGER.info("Blind category agreement (of confirmed hallucinations): %s",
               summary["stated_category_agreement_overall"])
    LOGGER.info("Unparseable rate: %.3f", summary["unparseable_rate"])
    LOGGER.info("Outputs: %s", out_dir)
    return 0


def run_agreement(paths: Sequence[str]) -> int:
    if len(paths) != 2:
        print("--agreement_between takes exactly two judgments_*.jsonl paths.", file=sys.stderr)
        return 2

    def load(path: str) -> List[Judgment]:
        out = []
        for row in load_jsonl(path):
            out.append(Judgment(row["benchmark_id"], row["dataset"], row["intended_category"],
                                row["is_hallucination"], row["stated_category"], "", "",
                                "", row.get("category_match")))
        return out

    a, b = load(paths[0]), load(paths[1])
    agreement = compute_agreement(a, b)
    print(json.dumps(agreement, indent=2))
    return 0


# ============================== self-test =================================

def self_test() -> int:
    fails = 0

    def check(name, cond, extra=""):
        nonlocal fails
        ok = bool(cond)
        fails += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'' if ok else '  ' + extra}")

    # answer parsing
    j = '{"is_hallucination": "true", "category": "relation_hallucination", ' \
        '"clinically_meaningful": "true", "unambiguous": "true", "rationale": "x"}'
    row = {"benchmark_id": "b1", "dataset": "DDI", "hallucination_category": "relation_hallucination"}
    jm = parse_judgment(row, j)
    check("parses well-formed judgment", jm.is_hallucination == "true"
          and jm.stated_category == "relation_hallucination" and jm.category_match is True)

    jm2 = parse_judgment(row, '<think>reasoning</think>' + j)
    check("strips reasoning before parsing", jm2.is_hallucination == "true")

    jm3 = parse_judgment(row, "garbage")
    check("unparseable response handled without crashing",
          jm3.is_hallucination == "unparseable" and jm3.category_match is None)

    row_wrong = {"benchmark_id": "b2", "dataset": "DDI", "hallucination_category": "incompleteness"}
    j_wrong_cat = '{"is_hallucination": "true", "category": "overgeneration", ' \
                  '"clinically_meaningful": "false", "unambiguous": "false", "rationale": "y"}'
    jm4 = parse_judgment(row_wrong, j_wrong_cat)
    check("category mismatch correctly flagged", jm4.category_match is False)

    row_clean = {"benchmark_id": "b3", "dataset": "DDI", "hallucination_category": "incompleteness"}
    j_clean = '{"is_hallucination": "false", "category": "none", ' \
              '"clinically_meaningful": "not_applicable", "unambiguous": "true", "rationale": "z"}'
    jm5 = parse_judgment(row_clean, j_clean)
    check("category_match is None when is_hallucination is false (nothing to compare)",
          jm5.category_match is None)

    # regression test for the real bug found in the Mistral-Nemo/Phi-4 runs:
    # a model saying is_hallucination=false while still naming a real
    # category (e.g. "incompleteness") must not have that category counted --
    # this is exactly what made incompleteness's per-category recall look
    # near-100% while semantic_validity_rate was near 0 on real data.
    row_incompl = {"benchmark_id": "b4", "dataset": "BIORED", "hallucination_category": "incompleteness"}
    j_contradiction = ('{"is_hallucination": "false", "category": "incompleteness", '
                       '"clinically_meaningful": "true", "unambiguous": "true", "rationale": "w"}')
    jm6 = parse_judgment(row_incompl, j_contradiction)
    check("is_hallucination=false forces stated_category to 'none' even if the model "
          "named a real category -- the exact contradiction found in real data",
          jm6.stated_category == "none", f"got {jm6.stated_category!r}")
    check("is_hallucination=false also forces clinically_meaningful to not_applicable",
          jm6.clinically_meaningful == "not_applicable")

    j_notsure_contradiction = ('{"is_hallucination": "not sure", "category": "overgeneration", '
                              '"clinically_meaningful": "true", "unambiguous": "false", "rationale": "w"}')
    jm7 = parse_judgment(row_incompl, j_notsure_contradiction)
    check("is_hallucination='not sure' also forces stated_category to 'none'",
          jm7.stated_category == "none", f"got {jm7.stated_category!r}")

    # stratified sampling
    fake_hallu = []
    for ds in ["CHEMPROT", "DDI", "DCE", "BIORED"]:
        for cat in CATEGORIES:
            n = 3 if (ds, cat) != ("BIORED", "context_induced") else 1
            for i in range(n):
                fake_hallu.append({"benchmark_id": f"{ds}_{cat}_{i}", "dataset": ds,
                                   "hallucination_category": cat})
    sample = build_sample(fake_hallu, sample_size=100, per_cell_max=0, seed=0)
    cells_present = {(r["dataset"], r["hallucination_category"]) for r in sample}
    check("stratified sample covers every (dataset, category) cell",
          len(cells_present) == 16, f"got {len(cells_present)} of 16")
    check("stratified sample does not exceed available instances",
          len(sample) == len(fake_hallu), f"{len(sample)} vs {len(fake_hallu)}")

    small = build_sample(fake_hallu, sample_size=8, per_cell_max=0, seed=0)
    small_cells = {(r["dataset"], r["hallucination_category"]) for r in small}
    check("small sample_size still spreads across cells rather than exhausting one",
          len(small_cells) == 8, f"got {len(small_cells)} distinct cells for n=8")

    capped = build_sample(fake_hallu, sample_size=100, per_cell_max=1, seed=0)
    check("per_cell_max is respected", len(capped) == 16, f"got {len(capped)}")

    # prompt safety
    unsafe_row = {"benchmark_id": "x", "dataset": "DDI", "text": "t", "entities": [], "relations": [],
                 "hallucination_category": "incompleteness", "hallucination_provenance": {"a": 1},
                 "source_benchmark_id": "s", "is_hallucination_benchmark": True,
                 "metadata": {"relation_label_inventory": {"EFFECT": 3}}}
    prompt, _ = build_comparison_prompt(unsafe_row, None)
    check("non-blind prompt DOES show intended_category (deliberately, by design -- "
          "the opposite of the earlier blind version)",
          "incompleteness" in prompt)
    check("prompt never leaks provenance/metadata/source-tracking keys regardless",
          "hallucination_provenance" not in prompt and "relation_label_inventory" not in prompt
          and "is_hallucination_benchmark" not in prompt)

    # ---- truncation: the actual bug that broke Mistral-Nemo/Yi/DeepSeek-67B ----
    huge_row = {"benchmark_id": "huge", "dataset": "DDI", "hallucination_category": "overgeneration",
               "text": "x" * 20000,
               "entities": [{"entity_id": f"e{i}", "type": "DRUG", "text": f"drug{i}"}
                            for i in range(80)],
               "relations": [{"label": "EFFECT", "arguments": [
                   {"role": "e1", "entity_id": f"e{i}"}, {"role": "e2", "entity_id": f"e{i+1}"}]}
                             for i in range(80)]}
    no_cap, flags_no_cap = render_comparison_record(huge_row, None, 0, 0, 0, 0)
    check("no truncation by default args (0 = uncapped) -- this is what just broke in prod",
          len(no_cap) > 20000 and not any(flags_no_cap.values()))

    capped, flags_capped = render_comparison_record(huge_row, None, 6000, 3000, 60, 60)
    check("capped prompt is dramatically smaller than uncapped",
          len(capped) < len(no_cap), f"{len(capped)} vs {len(no_cap)}")
    check("capped prompt's text field actually respects the char limit",
          len(capped) < len(no_cap) * 0.7, f"capped={len(capped)} uncapped={len(no_cap)}")
    check("truncation flags correctly report what was cut",
          flags_capped["text"] and flags_capped["entities"] and flags_capped["relations"])
    check("truncated payload tells the model it was truncated",
          "_validator_note" in capped)

    small_row = {"benchmark_id": "small", "dataset": "DDI", "hallucination_category": "overgeneration",
                "text": "short", "entities": [{"entity_id": "e0", "type": "DRUG", "text": "d"}],
                "relations": [{"label": "EFFECT", "arguments": [
                    {"role": "e1", "entity_id": "e0"}, {"role": "e2", "entity_id": "e0"}]}]}
    _, flags_small = render_comparison_record(small_row, None, 6000, 3000, 60, 60)
    check("a record well under the cap is never marked as truncated",
          not any(flags_small.values()))

    # metrics
    js = [
        Judgment("i1", "DDI", "relation_hallucination", "true", "relation_hallucination",
                "true", "true", "", True),
        Judgment("i2", "DDI", "relation_hallucination", "true", "overgeneration",
                "false", "true", "", False),
        Judgment("i3", "DDI", "incompleteness", "false", "none",
                "not_applicable", "true", "", None),
    ]
    s = summarize(js)
    check("semantic_validity_rate computed correctly",
          abs(s["semantic_validity_rate"] - round(2 / 3, 4)) < 1e-4)
    check("stated_category_agreement_overall excludes non-hallucination instances",
          abs(s["stated_category_agreement_overall"] - 0.5) < 1e-6)

    # kappa: perfect agreement -> 1.0; chance-level random labels on a
    # balanced 2-class set -> kappa near 0
    perfect = cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    check("perfect agreement -> kappa 1.0", perfect == 1.0)
    opposite = cohens_kappa(["a", "a", "a", "a"], ["b", "b", "b", "b"])
    check("kappa handles zero-overlap labels without dividing by zero", opposite == 0.0)

    ja = [Judgment("i1", "DDI", "relation_hallucination", "true", "relation_hallucination",
                   "", "", "", True),
          Judgment("i2", "DDI", "incompleteness", "true", "incompleteness", "", "", "", True)]
    jb = [Judgment("i2", "DDI", "incompleteness", "true", "overgeneration", "", "", "", False),
          Judgment("i1", "DDI", "relation_hallucination", "true", "relation_hallucination",
                   "", "", "", True)]
    agree = compute_agreement(ja, jb)
    check("cross-validator agreement matches by benchmark_id, not file order",
          agree["n_paired"] == 2 and agree["stated_category_percent_agreement"] == 0.5,
          str(agree))

    # ---- category order bias: the actual bug this run just uncovered ----
    fixed1 = category_block()
    fixed2 = category_block()
    check("no rng given -> deterministic fixed order (backward compatible)", fixed1 == fixed2)
    check("no rng given -> relation_hallucination listed first (old, biased default)",
          fixed1.strip().startswith("- relation_hallucination"))

    orders = {category_block(random.Random(f"seed{s}")) for s in range(30)}
    check("rng given -> order actually varies across instances", len(orders) > 1,
          f"only {len(orders)} distinct orderings across 30 seeds")

    first_word_counts = Counter()
    for s in range(200):
        blk = category_block(random.Random(f"seed{s}"))
        first_cat = blk.strip().split("\n")[0].split(":")[0].strip("- ").strip()
        first_word_counts[first_cat] += 1
    check("shuffled order is roughly uniform across which category leads "
          "(no residual first-listed bias)",
          max(first_word_counts.values()) < 200 * 0.40, dict(first_word_counts))

    # deterministic per source row: same benchmark_id -> same order, so a
    # rerun with the same seed reproduces the identical prompt.
    r1 = category_block(random.Random("42:row_abc"))
    r2 = category_block(random.Random("42:row_abc"))
    check("same seed string -> reproducible order for the same instance", r1 == r2)

    # ---- precision/recall exposes bias that raw accuracy hides ----
    # Simulate exactly what phi-4 did: guess relation_hallucination almost
    # always, regardless of the true category.
    biased_judgments = []
    for true_cat, n in [("relation_hallucination", 125), ("incompleteness", 125),
                        ("overgeneration", 125), ("context_induced", 125)]:
        for i in range(n):
            guess = true_cat if i < 5 else "relation_hallucination"  # 5 correct, rest biased
            biased_judgments.append(Judgment(
                f"{true_cat}_{i}", "DDI", true_cat, "true", guess, "", "true", "", guess == true_cat))
    bs = summarize(biased_judgments)
    pr = bs["per_category_precision_recall"]
    check("biased validator: relation_hallucination shows inflated recall",
          pr["relation_hallucination"]["recall"] > 0.9,
          str(pr["relation_hallucination"]))
    check("biased validator: relation_hallucination precision is NOT inflated "
          "(reveals the bias that recall alone hides)",
          pr["relation_hallucination"]["precision"] < 0.35,
          str(pr["relation_hallucination"]))
    check("biased validator: other categories show collapsed recall but real precision",
          pr["context_induced"]["recall"] < 0.1 and pr["context_induced"]["precision"] > 0.9,
          str(pr["context_induced"]))
    check("dominant-guess bias is flagged automatically",
          bs["_bias_warning"] is not None and bs["max_single_label_guess_share"] > 0.7,
          f"share={bs['max_single_label_guess_share']}")

    balanced_judgments = [
        Judgment(f"i{i}", "DDI", CATEGORIES[i % 4], "true", CATEGORIES[i % 4], "", "true", "", True)
        for i in range(100)
    ]
    bs2 = summarize(balanced_judgments)
    check("no bias warning when guesses are evenly distributed and correct",
          bs2["_bias_warning"] is None, str(bs2["max_single_label_guess_share"]))

    # ---- concurrent-launch safety ----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        state_path = str(Path(td) / "sample_ids.json")
        hallu_file = Path(td) / "hallu.jsonl"
        with open(hallu_file, "w", encoding="utf-8") as fh:
            for r in fake_hallu:
                fh.write(json.dumps(r) + "\n")

        class BuildArgs:
            hallu_jsonl = str(hallu_file)
            sample_state_file = state_path
            sample_size = 20
            per_cell_max = 0
            seed = 42
            output_dir = td
            sample_strategy = "equal"
            min_per_cell = 0
            export_human_review = False

        rc = run_build_sample_only(BuildArgs())
        check("--build_sample_only exits 0 and needs no model/backend", rc == 0)
        check("--build_sample_only actually wrote the file", Path(state_path).exists())
        saved_ids = json.load(open(state_path, encoding="utf-8"))
        check("--build_sample_only writes valid, non-empty id list", len(saved_ids) == 20)

        class RunArgs:
            hallu_jsonl = str(hallu_file)
            sample_state_file = str(Path(td) / "does_not_exist.json")
            require_prebuilt_sample = True
            sample_size = 20
            per_cell_max = 0
            seed = 42
            output_dir = td
            backend = "echo"
            validator_name = "t"
            batch_size = 512
            save_raw = False
            max_new_tokens = 400
            sample_strategy = "equal"
            min_per_cell = 0

        rc2 = run(RunArgs())
        check("--require_prebuilt_sample refuses to race-build a missing file",
              rc2 == 1)

    class BuildArgsNoState:
        sample_state_file = ""
    check("--build_sample_only without --sample_state_file is rejected",
          run_build_sample_only(BuildArgsNoState()) == 2)

    # ---- proportional apportionment ----
    sizes = {"a": 3020, "b": 875, "c": 364, "d": 62}  # real-ish, from DDI/CHEMPROT/DCE/BIORED overgeneration
    total = sum(sizes.values())
    alloc = apportion_proportional(sizes, 100)
    check("apportionment sums exactly to the requested total",
          sum(alloc.values()) == 100, str(alloc))
    check("largest real cell gets the largest allocation",
          alloc["a"] == max(alloc.values()), str(alloc))
    check("allocation never exceeds a cell's own availability",
          all(alloc[c] <= sizes[c] for c in sizes), str(alloc))
    expected_a = round(100 * sizes["a"] / total)
    check("largest cell's share is within 1 of its exact proportional share",
          abs(alloc["a"] - expected_a) <= 1, f"got {alloc['a']} expected ~{expected_a}")

    tiny = {"a": 1000, "b": 3}
    alloc_floor = apportion_proportional(tiny, 50, min_per_cell=10)
    check("min_per_cell floor rescues a cell too small to earn proportional seats",
          alloc_floor["b"] == 3, str(alloc_floor))  # capped by its own availability (only 3 exist)
    alloc_no_floor = apportion_proportional(tiny, 50, min_per_cell=0)
    check("without a floor, a tiny cell can be starved to zero",
          alloc_no_floor["b"] == 0, str(alloc_no_floor))

    check("total <= 0 returns all zeros without crashing",
          all(v == 0 for v in apportion_proportional(sizes, 0).values()))
    check("empty corpus returns all zeros without dividing by zero",
          all(v == 0 for v in apportion_proportional({"a": 0, "b": 0}, 10).values()))

    # ---- build_sample: proportional vs equal actually differ, as designed ----
    skewed_hallu = (
        [{"benchmark_id": f"big_{i}", "dataset": "DDI", "hallucination_category": "overgeneration"}
         for i in range(300)]
        + [{"benchmark_id": f"small_{i}", "dataset": "BIORED", "hallucination_category": "overgeneration"}
           for i in range(10)]
    )
    prop = build_sample(skewed_hallu, 50, 0, seed=0, strategy="proportional", min_per_cell=0)
    equal = build_sample(skewed_hallu, 50, 0, seed=0, strategy="equal", min_per_cell=0)
    prop_big = sum(1 for r in prop if r["dataset"] == "DDI")
    equal_big = sum(1 for r in equal if r["dataset"] == "DDI")
    check("proportional strategy weights the large cell much more heavily than equal does",
          prop_big > equal_big, f"proportional DDI={prop_big} equal DDI={equal_big}")
    check("equal strategy uses every available instance from the small cell",
          sum(1 for r in equal if r["dataset"] == "BIORED") == 10,
          f"equal BIORED={sum(1 for r in equal if r['dataset'] == 'BIORED')}")
    check("equal strategy then fills remaining quota from the only cell with supply left",
          equal_big == 40, f"equal DDI={equal_big}")

    # ---- composition report doesn't crash and mentions every cell ----
    report = sample_composition_report(prop, skewed_hallu)
    check("composition report includes both dataset cells",
          "DDI" in report and "BIORED" in report)

    # ---- human review export/import round-trip (single self-contained file) ----
    with tempfile.TemporaryDirectory() as td2:
        gold_h1 = {"benchmark_id": "DDI_999", "dataset": "DDI", "text": "Warfarin plus aspirin.",
                  "context_text": "", "entities": [
                      {"entity_id": "e0", "type": "DRUG", "text": "warfarin"},
                      {"entity_id": "e1", "type": "DRUG", "text": "aspirin"}],
                  "relations": [{"label": "EFFECT", "arguments": [
                      {"role": "e1", "entity_id": "e0"}, {"role": "e2", "entity_id": "e1"}]}]}
        mixed_sample = [
            {"benchmark_id": "h1", "dataset": "DDI", "hallucination_category": "relation_hallucination",
             "source_benchmark_id": "DDI_999",
             "text": "Warfarin plus aspirin.", "context_text": "", "entities": [
                 {"entity_id": "e0", "type": "DRUG", "text": "warfarin"},
                 {"entity_id": "e1", "type": "DRUG", "text": "aspirin"}],
             "relations": [{"label": "MECHANISM", "arguments": [
                 {"role": "e1", "entity_id": "e0"}, {"role": "e2", "entity_id": "e1"}]}],
             "hallucination_provenance": {"secret": "must never reach the CSV"},
             "metadata": {"relation_label_inventory": {"EFFECT": 1}}},
            {"benchmark_id": "h2", "dataset": "BIORED", "hallucination_category": "context_induced",
             "source_benchmark_id": "MISSING_SOURCE",
             "text": "Some abstract.", "context_text": "Some context.", "entities": [],
             "relations": [], "hallucination_provenance": {"secret2": "also must never leak"}},
        ]
        gold_idx0 = {"DDI_999": gold_h1}  # h2's source_benchmark_id deliberately unresolvable
        csv_path0, key_path = export_for_human_review(mixed_sample, td2, seed=0, gold_index=gold_idx0)
        check("export writes exactly ONE combined CSV, not one per cell",
              Path(csv_path0).exists() and Path(csv_path0).name == "human_review.csv")
        check("answer key file also exists, as a redundant backup",
              Path(key_path).exists() and "DO_NOT_SHARE" in key_path)

        with open(csv_path0, encoding="utf-8") as fh:
            content = fh.read()
        check("exported CSV never leaks hallucination_provenance",
              "must never reach the CSV" not in content)
        with open(csv_path0, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        check("the single file contains rows for every cell in the sample, not just one",
              {r["benchmark_id"] for r in rows} == {"h1", "h2"})
        row_h1 = next(r for r in rows if r["benchmark_id"] == "h1")
        row_h2 = next(r for r in rows if r["benchmark_id"] == "h2")
        check("CSV is self-contained: benchmark_id is its own column, no answer key needed to read it",
              row_h1["benchmark_id"] == "h1" and row_h1["dataset"] == "DDI")
        check("CSV is self-contained: intended_category is its own column too",
              row_h1["intended_category"] == "relation_hallucination")
        check("exported row shows GOLD relations distinctly from MODIFIED",
              "EFFECT" in row_h1["gold_relations"] and "MECHANISM" in row_h1["modified_relations"])
        check("exported row includes a readable change_summary, not raw diff internals",
              "LABEL CHANGED" in row_h1["change_summary"])
        check("blank judgment columns are present and empty, ready for a WebForm response",
              row_h1["is_hallucination"] == "" and row_h1["category"] == "")
        check("a row whose source_benchmark_id has no match exports the modified-only "
              "fallback instead of crashing",
              "unavailable" in row_h2["gold_relations"].lower())

        # simulate a WebForm response coming back as a single completed CSV
        for r in rows:
            r["is_hallucination"] = "true"
            r["category"] = "relation_hallucination"  # deliberately WRONG for h2, correct for h1
            r["clinically_meaningful"] = "true"
            r["unambiguous"] = "true"
            r["rationale"] = "test rationale"
        with open(csv_path0, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=HUMAN_REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        # the actual point of this change: load with NO answer key at all
        judgments, n_incomplete = load_human_review([csv_path0])
        check("completed single-file review round-trips with NO answer_key argument at all",
              len(judgments) == 2, f"got {len(judgments)}")
        check("n_incomplete is 0 once every row has been filled in", n_incomplete == 0)
        by_id = {j.benchmark_id: j for j in judgments}
        check("round-tripped judgment correctly matches benchmark_id back to the real source row",
              set(by_id) == {"h1", "h2"})
        check("category_match is computed correctly against the TRUE intended category, "
              "not just echoed back",
              by_id["h1"].category_match is True and by_id["h2"].category_match is False,
              f"h1={by_id['h1'].category_match} h2={by_id['h2'].category_match}")

        # answer key still works as a fallback for a row missing the new columns
        # (e.g. an old export, or a form tool that stripped a column)
        legacy_path = Path(td2) / "legacy_export.csv"
        with open(legacy_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["review_id", "text", "is_hallucination", "category",
                             "clinically_meaningful", "unambiguous", "rationale"])
            writer.writerow(["R99999", "t", "true", "incompleteness", "true", "true", "done"])
            writer.writerow(["R99998", "t", "", "", "", "", ""])
        legacy_key = {"R99999": {"benchmark_id": "h3", "dataset": "TEST",
                                 "intended_category": "incompleteness"},
                     "R99998": {"benchmark_id": "h4", "dataset": "TEST",
                               "intended_category": "incompleteness"}}
        legacy_key_path = Path(td2) / "legacy_key.json"
        legacy_key_path.write_text(json.dumps(legacy_key), encoding="utf-8")
        j2, incomplete2 = load_human_review([str(legacy_path)], str(legacy_key_path))
        check("a row missing the self-contained columns falls back to the answer key correctly",
              len(j2) == 1 and j2[0].benchmark_id == "h3")
        check("a still-blank row is excluded from judgments, not counted as 'unparseable'",
              len(j2) == 1)
        check("a still-blank row IS counted in n_incomplete", incomplete2 == 1, f"got {incomplete2}")

    # ---- comparative (non-blind) rendering, diff, and prompt construction ----
    # ---- comparative (non-blind) rendering, diff, and prompt construction ----
    # This is a direct regression test for the two crash bugs found in this
    # design: category_block was referenced by build_comparison_prompt but
    # never defined, and render_comparison_record's gold-missing fallback
    # called a function (render_record) that did not exist anywhere in the
    # file. Both would only surface at the moment these functions actually
    # ran -- exactly what this block now exercises end to end.
    gold_row = {
        "benchmark_id": "BIORED_999", "dataset": "BIORED", "text": "Gene X is linked to disease Y.",
        "context_text": "Background on gene X.",
        "entities": [{"entity_id": "g1", "type": "GeneOrGeneProduct", "text": "X"},
                    {"entity_id": "d1", "type": "DiseaseOrPhenotypicFeature", "text": "Y"}],
        "relations": [{"relation_id": "R0", "label": "Positive_Correlation",
                      "arguments": [{"role": "entity1", "entity_id": "g1"},
                                   {"role": "entity2", "entity_id": "d1"}]}],
    }
    modified_row = {
        "benchmark_id": "BIORED_999_HAL_relation_hallucination", "dataset": "BIORED",
        "hallucination_category": "relation_hallucination", "source_benchmark_id": "BIORED_999",
        "text": gold_row["text"], "context_text": gold_row["context_text"],
        "entities": gold_row["entities"],
        "relations": [{"relation_id": "H0", "label": "Negative_Correlation",
                      "arguments": [{"role": "entity1", "entity_id": "g1"},
                                   {"role": "entity2", "entity_id": "d1"}]}],
    }

    check("category_block is defined and returns all four categories",
          all(c in category_block() for c in CATEGORIES))
    order_varies = {category_block(random.Random(s)) for s in range(20)}
    check("category_block order varies when given an rng (kills first-listed bias)",
          len(order_varies) > 1, f"{len(order_varies)} distinct orderings across 20 seeds")

    rendered, cflags = render_comparison_record(modified_row, gold_row)
    obj = json.loads(rendered)
    check("comparative rendering shows intended_category directly, not hidden",
          obj.get("intended_category") == "relation_hallucination")
    check("comparative rendering shows GOLD relations", "Positive_Correlation" in obj["gold_relations"])
    check("comparative rendering shows MODIFIED relations distinctly from gold",
          "Negative_Correlation" in obj["modified_relations"])
    check("comparative rendering's change_summary correctly identifies a label change, "
          "not a spurious remove+add pair",
          "LABEL CHANGED" in obj["change_summary"]
          and "RELATION REMOVED" not in obj["change_summary"]
          and "RELATION ADDED" not in obj["change_summary"],
          obj["change_summary"])
    check("unchanged context is shown once (context_text), not duplicated as two identical fields",
          "context_text" in obj and "original_context_text" not in obj)

    # a context_induced case where context genuinely differs
    ctx_gold = dict(gold_row)
    ctx_mod = dict(modified_row)
    ctx_mod["context_text"] = "A rewritten passage implying an unsupported link."
    ctx_mod["hallucination_category"] = "context_induced"
    rendered2, _ = render_comparison_record(ctx_mod, ctx_gold)
    obj2 = json.loads(rendered2)
    check("rewritten context surfaces BOTH versions with a note, not just the new one",
          obj2.get("original_context_text") == gold_row["context_text"]
          and obj2.get("modified_context_text") == ctx_mod["context_text"])

    # missing gold row falls back gracefully rather than crashing (the second
    # bug this block exists to catch)
    rendered3, _ = render_comparison_record(modified_row, None)
    obj3 = json.loads(rendered3)
    check("a missing gold row falls back to the modified-only view with an explanatory note, "
          "instead of raising NameError from a nonexistent render_record",
          "_comparison_note" in obj3 and "modified_relations" in obj3
          and "gold_relations" not in obj3)

    # the comparative prompt is genuinely not blind: it explicitly shows both
    # sets, plus intended_category, and this call itself is the direct
    # regression test for the missing-category_block crash.
    comp_prompt, _ = build_comparison_prompt(modified_row, gold_row)
    check("comparative prompt literally contains both GOLD and MODIFIED relation content",
          "Positive_Correlation" in comp_prompt and "Negative_Correlation" in comp_prompt)
    check("comparative prompt shows intended_category rather than hiding it",
          "relation_hallucination" in comp_prompt)
    check("comparative prompt instructs the reviewer to compare gold vs modified, not guess blind",
          "GOLD" in comp_prompt and "MODIFIED" in comp_prompt)

    # comparative export: CSV shows gold and modified side by side
    with tempfile.TemporaryDirectory() as td3:
        gold_idx = {"BIORED_999": gold_row}
        csv_path3, key3 = export_for_human_review([modified_row], td3, seed=0, gold_index=gold_idx)
        with open(csv_path3, encoding="utf-8", newline="") as fh:
            rows3 = list(csv.DictReader(fh))
        check("comparative CSV export has separate gold_relations and modified_relations columns",
              "Positive_Correlation" in rows3[0]["gold_relations"]
              and "Negative_Correlation" in rows3[0]["modified_relations"])
        check("comparative CSV export includes a readable change_summary column",
              "LABEL CHANGED" in rows3[0]["change_summary"])
        check("comparative CSV export shows intended_category as its own column",
              rows3[0]["intended_category"] == "relation_hallucination")

        # missing gold in export path: should not crash, should say so
        csv_path4, key4 = export_for_human_review([modified_row], td3, seed=1, gold_index={})
        with open(csv_path4, encoding="utf-8", newline="") as fh:
            rows4 = list(csv.DictReader(fh))
        check("comparative export with no matching gold row explains the gap instead of crashing",
              "unavailable" in rows4[0]["gold_relations"].lower())

    check("index_by_benchmark_id builds a correct lookup", index_by_benchmark_id(
        [gold_row])["BIORED_999"]["dataset"] == "BIORED")

    # ---- verdict logic ----
    def mkj(bid, ds, cat, is_h):
        return Judgment(bid, ds, cat, is_h, cat if is_h == "true" else "none",
                        "true", "true", "", (is_h == "true"))

    # cell 1: both validators agree true on all 10 -> accept
    # cell 2: validators disagree constantly -> low kappa -> not accept
    # cell 3: both agree false mostly -> low consensus -> regenerate
    val_a, val_b = [], []
    for i in range(10):
        val_a.append(mkj(f"c1_{i}", "DDI", "relation_hallucination", "true"))
        val_b.append(mkj(f"c1_{i}", "DDI", "relation_hallucination", "true"))
    for i in range(10):
        va = "true" if i % 2 == 0 else "false"
        vb = "false" if i % 2 == 0 else "true"
        val_a.append(mkj(f"c2_{i}", "DDI", "overgeneration", va))
        val_b.append(mkj(f"c2_{i}", "DDI", "overgeneration", vb))
    for i in range(10):
        val_a.append(mkj(f"c3_{i}", "BIORED", "context_induced", "false"))
        val_b.append(mkj(f"c3_{i}", "BIORED", "context_induced", "false"))

    verdict = compute_verdict({"model_a": val_a, "model_b": val_b})
    check("high-agreement, high-validity cell is accepted",
          verdict["cells"]["DDI:relation_hallucination"]["verdict"] == "accept",
          str(verdict["cells"]["DDI:relation_hallucination"]))
    check("chance-level-disagreement cell is NOT accepted (low kappa catches it "
          "even though raw validity numbers could look moderate)",
          verdict["cells"]["DDI:overgeneration"]["verdict"] != "accept",
          str(verdict["cells"]["DDI:overgeneration"]))
    check("consistently-false cell (low consensus validity) is flagged for regeneration",
          verdict["cells"]["BIORED:context_induced"]["verdict"] == "regenerate",
          str(verdict["cells"]["BIORED:context_induced"]))
    check("verdict_counts tally matches the three cells",
          sum(verdict["verdict_counts"].values()) == 3, str(verdict["verdict_counts"]))

    try:
        compute_verdict({"only_one": val_a})
        check("compute_verdict rejects fewer than 2 validators", False)
    except ValueError:
        check("compute_verdict rejects fewer than 2 validators", True)

    # ---- the actual scenario requested: 2 LLM validators + human, merged ----
    # Deliberately disagreeing in a way that requires majority vote (2-of-3),
    # not unanimity, and mixes a model's judgments with a human's read
    # straight from a completed CSV to prove the whole path -- WebForm export
    # to load_human_review to compute_verdict -- works end to end together.
    model_a, model_b, human = [], [], []
    for i in range(9):
        model_a.append(mkj(f"m1_{i}", "DDI", "relation_hallucination", "true"))
        model_b.append(mkj(f"m1_{i}", "DDI", "relation_hallucination", "true"))
        # human disagrees on 1 of 9 -- majority (2-of-3) still says true
        human.append(mkj(f"m1_{i}", "DDI", "relation_hallucination",
                         "false" if i == 0 else "true"))
    three_way = compute_verdict({"model_a": model_a, "model_b": model_b, "human": human})
    check("three-way verdict (2 models + human) uses 2-of-3 majority, not unanimity",
          three_way["cells"]["DDI:relation_hallucination"]["consensus_validity_rate"] == 1.0,
          str(three_way["cells"]["DDI:relation_hallucination"]))
    check("three-way verdict records all three validator names",
          three_way["validators"] == ["human", "model_a", "model_b"], str(three_way["validators"]))
    check("mean pairwise kappa correctly stays LOW here despite 8/9 raw agreement: kappa "
          "degenerates to 0 between any rater and a near-constant one (human vs. either "
          "all-true model), which is real Cohen's-kappa behaviour, not a bug -- this is "
          "exactly the case where trusting raw agreement instead of kappa would overstate "
          "reliability, and why the verdict lands on adjudicate rather than accept",
          three_way["cells"]["DDI:relation_hallucination"]["verdict"] == "adjudicate",
          str(three_way["cells"]["DDI:relation_hallucination"]))

    # for contrast: full three-way agreement with no disagreement at all
    # should clear both thresholds and reach accept
    full_agree = {"model_a": [], "model_b": [], "human": []}
    for i in range(9):
        for name in full_agree:
            full_agree[name].append(mkj(f"fa_{i}", "DDI", "relation_hallucination", "true"))
    three_way_clean = compute_verdict(full_agree)
    check("three-way verdict WITH true full agreement (no disagreement at all) reaches accept",
          three_way_clean["cells"]["DDI:relation_hallucination"]["verdict"] == "accept",
          str(three_way_clean["cells"]["DDI:relation_hallucination"]))

    # now round-trip the "human" leg through an actual WebForm-style CSV
    # export/import, not just a hand-built Judgment list, to prove the full
    # chain -- export, human fills it in, import, verdict -- works together.
    with tempfile.TemporaryDirectory() as td4:
        human_sample = [
            {"benchmark_id": f"m1_{i}", "dataset": "DDI",
             "hallucination_category": "relation_hallucination",
             "source_benchmark_id": None,
             "text": "t", "context_text": "", "entities": [], "relations": []}
            for i in range(9)
        ]
        csv_path5, _ = export_for_human_review(human_sample, td4, seed=0, gold_index={})
        with open(csv_path5, encoding="utf-8", newline="") as fh:
            rows5 = list(csv.DictReader(fh))
        for r in rows5:
            is_first = r["benchmark_id"] == "m1_0"
            r["is_hallucination"] = "false" if is_first else "true"
            r["category"] = "none" if is_first else "relation_hallucination"
            r["clinically_meaningful"] = "not_applicable" if is_first else "true"
            r["unambiguous"] = "true"
            r["rationale"] = "webform test"
        with open(csv_path5, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=HUMAN_REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows5)
        human_from_csv, _ = load_human_review([csv_path5])  # no answer_key at all
        three_way_real = compute_verdict({"model_a": model_a, "model_b": model_b,
                                          "human": human_from_csv})
        check("full chain (export -> WebForm fill-in -> import -> verdict) matches "
              "the hand-built version above exactly",
              three_way_real["cells"]["DDI:relation_hallucination"]["verdict"]
              == three_way["cells"]["DDI:relation_hallucination"]["verdict"]
              == "adjudicate",
              str(three_way_real["cells"]["DDI:relation_hallucination"]))

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


# =================================== CLI ===================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent validation of re_hallu_benchmark.jsonl (not model detection perf).",
        allow_abbrev=False)
    p.add_argument("--hallu_jsonl", type=str, default="")
    p.add_argument("--unified_jsonl", type=str, default="",
                   help="Required: the source of the gold relation sets shown alongside each "
                        "modified (hallucinated) one, matched via source_benchmark_id. The "
                        "reviewer -- model or human -- sees GOLD and MODIFIED relations side "
                        "by side (and both versions of context_text when it was rewritten), "
                        "plus intended_category and a readable change summary, and judges the "
                        "specific change rather than the modified record in isolation.")
    p.add_argument("--output_dir", type=str, default="outputs/validation")
    p.add_argument("--validator_name", type=str, default="",
                   help="Short label used in output filenames, e.g. 'llama3.3-70b', 'gpt41mini'.")

    p.add_argument("--sample_size", type=int, default=1000)
    p.add_argument("--per_cell_max", type=int, default=0, help="0 = no per-cell cap.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_strategy", type=str, default="proportional",
                   choices=["proportional", "equal"],
                   help="'proportional' (default): each (dataset, category) cell's sample "
                        "size mirrors its real share of the corpus, so the sample's "
                        "composition matches the benchmark as shipped. 'equal': the "
                        "original behaviour, every cell gets ~sample_size/16 regardless of "
                        "its real size -- use this only to reproduce an existing sample "
                        "built before this flag existed. Changing strategy on an existing "
                        "--sample_state_file path invalidates any prior --agreement_between "
                        "comparison against it; use a new path.")
    p.add_argument("--min_per_cell", type=int, default=20,
                   help="Floor applied after proportional allocation so a cell too small to "
                        "earn seats proportionally still gets checked at all. 0 disables. "
                        "Only used with --sample_strategy proportional.")
    p.add_argument("--sample_state_file", type=str, default="",
                   help="Save/reuse the exact sampled benchmark_ids here, so a second "
                        "validator judges the identical instances (required for kappa).")
    p.add_argument("--build_sample_only", action="store_true",
                   help="Build and save --sample_state_file, then exit. No model, no "
                        "backend, no GPU. Run this once before launching validators in "
                        "parallel -- see module docstring for why concurrent creation "
                        "from validator processes themselves is unsafe.")
    p.add_argument("--require_prebuilt_sample", action="store_true",
                   help="Refuse to build the sample inline if --sample_state_file is "
                        "missing; error out instead, so a mistimed parallel launch fails "
                        "loudly instead of silently racing. Recommended whenever you "
                        "intend to run validators concurrently.")
    p.add_argument("--export_human_review", action="store_true",
                   help="Also (with --build_sample_only) or instead (standalone) write "
                        "blind per-(dataset,category) CSVs for human annotators, plus a "
                        "separately-named answer key. See --import_human_review to bring "
                        "completed reviews back in.")
    p.add_argument("--import_human_review", type=str, nargs="+", default=None,
                   help="One or more completed human-review CSVs (from --export_human_review) "
                        "to parse into judgments_<name>.jsonl / summary_<name>.json, in the "
                        "same format the model validators produce. Requires --answer_key.")
    p.add_argument("--answer_key", type=str, default="",
                   help="Path to the _answer_key_DO_NOT_SHARE.json produced alongside the "
                        "human-review CSVs. Required with --import_human_review.")
    p.add_argument("--human_validator_name", type=str, default="human",
                   help="Label used in output filenames for --import_human_review, e.g. "
                        "judgments_human.jsonl.")
    p.add_argument("--verdict", type=str, nargs="+", default=None,
                   help="Two or more judgments_*.jsonl files (any mix of model and human "
                        "validators). Computes and prints the per-cell accept/adjudicate/ "
                        "regenerate verdict. No model calls, no sampling.")
    p.add_argument("--validity_thresholds", type=str, default="0.70,0.40",
                   help="accept,adjudicate consensus-validity cutoffs for --verdict.")
    p.add_argument("--kappa_thresholds", type=str, default="0.40,0.20",
                   help="accept,adjudicate mean-pairwise-kappa cutoffs for --verdict.")

    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "api", "echo"])
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--max_text_chars", type=int, default=6000,
                   help="Cap on the TEXT field before rendering. 0 = no cap. Default keeps "
                        "even the largest real abstracts (BioRED, ~7k measured max tokens "
                        "pre-truncation) safely under an 8192 context window with the "
                        "instruction block and --max_new_tokens included.")
    p.add_argument("--max_context_chars", type=int, default=3000)
    p.add_argument("--max_entities", type=int, default=60,
                   help="0 = no cap. One real DDI sentence in this corpus has 55 entities; "
                        "60 covers it with margin without needing this in the common case.")
    p.add_argument("--max_relations", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=512)

    p.add_argument("--api_model", type=str, default="")
    p.add_argument("--api_base_url", type=str, default="")
    p.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--api_concurrency", type=int, default=8)
    p.add_argument("--api_max_retries", type=int, default=5)
    p.add_argument("--api_request_timeout", type=float, default=60.0)

    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--agreement_between", type=str, nargs=2, default=None,
                   help="Two judgments_*.jsonl files; print Cohen's kappa between them "
                        "and exit. No model calls.")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.self_test:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return self_test()
    if args.agreement_between:
        return run_agreement(args.agreement_between)
    if args.verdict:
        if len(args.verdict) < 2:
            print("--verdict needs at least 2 judgments_*.jsonl files.", file=sys.stderr)
            return 2
        vt = tuple(float(x) for x in args.validity_thresholds.split(","))
        kt = tuple(float(x) for x in args.kappa_thresholds.split(","))
        return run_verdict(args.verdict, vt, kt)
    if args.import_human_review:
        return run_import_human_review(args)
    if not args.hallu_jsonl:
        print("--hallu_jsonl is required (or --self_test / --agreement_between / "
              "--verdict / --import_human_review).", file=sys.stderr)
        return 2
    if args.build_sample_only:
        return run_build_sample_only(args)
    if args.export_human_review:
        return run_export_human_review(args)
    if args.backend == "vllm" and not args.model_path:
        print("--model_path is required for --backend vllm.", file=sys.stderr)
        return 2
    if args.backend == "api" and not (args.api_model or args.model_path):
        print("--api_model is required for --backend api.", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/validate_re_hallu.py  --hallu_jsonl data/re_hallu_benchmark.jsonl --unified_jsonl data/unified_RE_benchmark.jsonl  --sample_size 2000 --min_per_cell 20  --sample_state_file outputs/validation/sample_ids.json --output_dir outputs/validation  --build_sample_only --export_human_review