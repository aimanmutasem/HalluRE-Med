#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("hallu_detect")

CATEGORIES = ["relation_hallucination", "incompleteness",
              "overgeneration", "context_induced"]

PROMPT_UNSAFE_KEYS = (
    "hallucination_provenance",
    "hallucination_category",
    "is_hallucination_benchmark",
    "source_benchmark_id",
    "metadata",
)

NOT_SURE = "not sure"


# =====================================================================
# IO
# =====================================================================

def setup_logging(output_dir: str, level: str = "INFO") -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(output_dir, "detection.log"),
                                      encoding="utf-8")],
        force=True)


def load_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def save_jsonl(rows: Sequence[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_reasoning(text: str) -> str:
    """Remove Qwen3 <think> ... </think> blocks before JSON extraction."""
    if "<think>" not in text:
        return text
    while "<think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            return text[:start]
        text = text[:start] + text[end + len("</think>"):]
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
    """Remove every field that would leak the answer to the evaluated model."""
    clean = copy.deepcopy(row)
    for key in PROMPT_UNSAFE_KEYS:
        clean.pop(key, None)
    return clean


# =====================================================================
# Evaluation-set construction
# =====================================================================

@dataclass
class Instance:
    instance_id: str
    source_id: str          # clustering unit for the bootstrap
    dataset: str
    record: Dict[str, Any]  # already stripped
    is_hallucinated: bool
    category: Optional[str]
    corruption_rate: Optional[float]
    gold_relation_count: int


def build_eval_set(hallu_rows: Sequence[Dict[str, Any]],
                   unified_rows: Sequence[Dict[str, Any]],
                   negative_ratio: float, seed: int,
                   max_per_cell: Optional[int] = None) -> List[Instance]:
    """Positives from the hallucination benchmark, negatives from untouched gold.

    Negatives are drawn per dataset so that class priors are balanced within
    each corpus rather than only globally - otherwise DDI (the largest slice)
    dominates and per-dataset accuracy is not comparable.
    """
    rng = random.Random(seed)
    positives: List[Instance] = []
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in hallu_rows:
        by_cell[(r.get("dataset", "?"), r.get("hallucination_category", "?"))].append(r)

    for (ds, cat), rows in by_cell.items():
        rows = list(rows)
        rng.shuffle(rows)
        if max_per_cell:
            rows = rows[:max_per_cell]
        for r in rows:
            prov = r.get("hallucination_provenance") or {}
            positives.append(Instance(
                instance_id=r.get("benchmark_id", ""),
                source_id=r.get("source_benchmark_id", r.get("benchmark_id", "")),
                dataset=ds,
                record=strip_prompt_unsafe(r),
                is_hallucinated=True,
                category=cat,
                corruption_rate=prov.get("corruption_rate"),
                gold_relation_count=prov.get("source_positive_relation_count", 0),
            ))

    pos_per_ds = Counter(i.dataset for i in positives)
    unified_by_ds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in unified_rows:
        if r.get("relations"):
            unified_by_ds[r.get("dataset", "?")].append(r)

    negatives: List[Instance] = []
    for ds, n_pos in pos_per_ds.items():
        pool = list(unified_by_ds.get(ds, []))
        rng.shuffle(pool)
        want = int(round(n_pos * negative_ratio))
        for r in pool[:want]:
            negatives.append(Instance(
                instance_id=f"{r.get('benchmark_id','')}_NEG",
                source_id=r.get("benchmark_id", ""),
                dataset=ds,
                record=strip_prompt_unsafe(r),
                is_hallucinated=False,
                category=None,
                corruption_rate=0.0,
                gold_relation_count=len([x for x in r.get("relations", [])
                                         if x.get("label") != "NON"]),
            ))
        if len(pool) < want:
            LOGGER.warning("%s: only %d gold rows available for %d requested negatives.",
                           ds, len(pool), want)

    allinst = positives + negatives
    rng.shuffle(allinst)
    return allinst


# =====================================================================
# Prompting
# =====================================================================

# =====================================================================
# External prompt files (--prompt_dir) -- REQUIRED. This module contains
# no prompt text of its own (matching run_e2e_llm.py's convention: all
# task instructions live in files, code only assembles them). The one
# narrow exception is --prompt_variant priority, a deliberately code-only
# ablation that reproduces the historical hard-coded prior for controlled
# comparison -- kept in code specifically so it can never drift from what
# it's supposed to be reproducing; see build_type_prompt.
#
# Required files in --prompt_dir:
#   system.txt                  system prompt
#   binary_instruction.txt      binary task instruction
#   type_instruction_head.txt   type task instruction, before the category list
#   type_instruction_tail.txt   type task instruction, after the category list
#   categories.json             {category_name: description} for all 4 CATEGORIES
# Optional, per dataset (falls back to no extra grounding if absent):
#   CHEMPROT.txt / DDI.txt / DCE.txt / BIORED.txt
#
# Every loaded file/value is scanned by check_for_bias() before use. A
# match refuses the run with a specific, actionable error rather than
# silently stripping the text or using it anyway -- silently rewriting a
# research prompt is its own integrity problem.

# Required file: system.txt, containing five sections delimited by
# "### NAME ###" marker lines:
#   SYSTEM               system prompt
#   BINARY_INSTRUCTION   binary task instruction
#   TYPE_INSTRUCTION_HEAD   type task instruction, before the category list
#   TYPE_INSTRUCTION_TAIL   type task instruction, after the category list
#   CATEGORIES            JSON object: {category_name: description} for all 4 CATEGORIES
# Optional, per dataset (falls back to no extra grounding if absent):
#   CHEMPROT.txt / DDI.txt / DCE.txt / BIORED.txt
#
# Every section's text is scanned by check_for_bias() before use. A match
# refuses the run with a specific, actionable error rather than silently
# stripping the text or using it anyway -- silently rewriting a research
# prompt is its own integrity problem.

REQUIRED_SECTIONS = ["SYSTEM", "BINARY_INSTRUCTION", "TYPE_INSTRUCTION_HEAD",
                     "TYPE_INSTRUCTION_TAIL", "CATEGORIES"]

SECTION_MARKER = re.compile(r"^###\s*([A-Z_]+)\s*###\s*$", re.MULTILINE)

BIAS_PATTERNS = [
    (re.compile(r"strict\s+priority", re.IGNORECASE),
     "a hard-coded 'Strict priority' ranking"),
    (re.compile(r"prefer\s+relation[_ ]hallucination", re.IGNORECASE),
     "'prefer relation_hallucination' as a default-uncertainty answer"),
    (re.compile(r"strongest\s+signal", re.IGNORECASE),
     "'(strongest signal)' priority framing"),
    (re.compile(r"^\s*1\.\s*relation_hallucination", re.IGNORECASE | re.MULTILINE),
     "a numbered ranking with relation_hallucination listed first"),
]


def check_for_bias(text: str, source_name: str) -> None:
    hits = [desc for pattern, desc in BIAS_PATTERNS if pattern.search(text)]
    if hits:
        raise ValueError(
            f"Refusing to load {source_name}: contains {', '.join(hits)}.\n"
            "This is the exact hard-coded label prior documented in this script's own "
            "docstring as a bug: it privileges relation_hallucination over the other "
            "three categories whenever a model is uncertain, and reproduces the "
            "'relation hallucination easiest, incompleteness hardest' finding as a "
            "prompt artifact rather than a measured result. Edit the file to remove "
            "this before proceeding. If you specifically want to reproduce the old "
            "prior as a labeled ablation, use --prompt_variant priority instead -- "
            "that path is code-only and does not read prompt files at all."
        )


def parse_sectioned_file(text: str, source_name: str) -> Dict[str, str]:
    """Split system.txt on '### NAME ###' marker lines into a
    {section_name: text} dict. Every REQUIRED_SECTIONS entry must be
    present exactly once; anything else in the file is an error rather
    than silently ignored, so a typo'd marker doesn't quietly drop a
    section back to an empty string.
    """
    markers = list(SECTION_MARKER.finditer(text))
    if not markers:
        raise ValueError(
            f"{source_name} has no '### NAME ###' section markers. Expected sections: "
            + ", ".join(REQUIRED_SECTIONS))

    sections: Dict[str, str] = {}
    for i, m in enumerate(markers):
        name = m.group(1)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        if name in sections:
            raise ValueError(f"{source_name} defines section '{name}' more than once.")
        sections[name] = text[start:end].strip()

    missing = [s for s in REQUIRED_SECTIONS if s not in sections]
    if missing:
        raise ValueError(f"{source_name} is missing required section(s): {missing}")
    extra = [s for s in sections if s not in REQUIRED_SECTIONS]
    if extra:
        raise ValueError(f"{source_name} has unrecognised section(s): {extra}. "
                         f"Expected only: {REQUIRED_SECTIONS}")
    return sections


@dataclass
class PromptBundle:
    system_prompt: str
    binary_instruction: str
    type_head: str
    type_tail: str
    category_gloss: Dict[str, str]
    dataset_guidance: Dict[str, str]   # dataset -> extra grounding text, "" if none supplied
    source: str


def load_prompt_bundle(prompt_dir: str) -> PromptBundle:
    if not prompt_dir:
        raise ValueError(
            "--prompt_dir is required: this script has no built-in prompt text. Point "
            "it at a directory containing system.txt (five '### NAME ###'-delimited "
            "sections: " + ", ".join(REQUIRED_SECTIONS) + "), and optionally "
            "CHEMPROT.txt/DDI.txt/DCE.txt/BIORED.txt for extra per-dataset grounding."
        )
    pdir = Path(prompt_dir)
    sys_path = pdir / "system.txt"
    if not sys_path.exists():
        raise ValueError(f"--prompt_dir {pdir} is missing required file: system.txt")

    raw = sys_path.read_text(encoding="utf-8")
    sections = parse_sectioned_file(raw, str(sys_path))
    for name, text in sections.items():
        check_for_bias(text, f"{sys_path} [{name}]")

    category_gloss = json.loads(sections["CATEGORIES"])
    if set(category_gloss) != set(CATEGORIES):
        raise ValueError(f"{sys_path} [CATEGORIES] must define exactly these categories: "
                         f"{CATEGORIES}; got {sorted(category_gloss)}")

    dataset_guidance: Dict[str, str] = {}
    for ds in ["CHEMPROT", "DDI", "DCE", "BIORED"]:
        p = pdir / f"{ds}.txt"
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            check_for_bias(text, str(p))
            dataset_guidance[ds] = text
        else:
            LOGGER.warning("%s not found; no extra dataset guidance for %s.", p, ds)

    return PromptBundle(sections["SYSTEM"], sections["BINARY_INSTRUCTION"],
                        sections["TYPE_INSTRUCTION_HEAD"], sections["TYPE_INSTRUCTION_TAIL"],
                        category_gloss, dataset_guidance, source=str(pdir))


def build_binary_prompt(inst: Instance, with_text: bool, bundle: PromptBundle,
                        max_text_chars: int = 0, max_context_chars: int = 0,
                        max_entities: int = 0, max_relations: int = 0
                        ) -> Tuple[str, Dict[str, bool]]:
    guidance = bundle.dataset_guidance.get(inst.dataset, "")
    rec, flags = render_record(inst.record, with_text, dataset_guidance=guidance,
                               max_text_chars=max_text_chars, max_context_chars=max_context_chars,
                               max_entities=max_entities, max_relations=max_relations)
    return f"{bundle.binary_instruction}\n\nRECORD:\n{rec}", flags


def build_type_prompt(inst: Instance, with_text: bool, variant: str,
                      rng: random.Random, bundle: PromptBundle,
                      max_text_chars: int = 0, max_context_chars: int = 0,
                      max_entities: int = 0, max_relations: int = 0
                      ) -> Tuple[str, Dict[str, bool]]:
    cats = list(CATEGORIES)
    if variant == "neutral":
        rng.shuffle(cats)                      # kill positional bias
    lines = [bundle.type_head]
    for c in cats:
        lines.append(f"- {c}: {bundle.category_gloss[c]}")
    if variant == "priority":
        # ABLATION ONLY - reproduces the prior baked into the old prompt.
        # Deliberately code-only: see the module note above for why.
        lines.append("\nStrict priority:")
        lines.append("1. relation_hallucination (strongest signal)")
        lines.append("2. overgeneration")
        lines.append("3. incompleteness")
        lines.append("4. context_induced")
        lines.append("If unsure whether a relation is correct -> prefer relation_hallucination.")
    lines.append(bundle.type_tail)
    guidance = bundle.dataset_guidance.get(inst.dataset, "")
    rec, flags = render_record(inst.record, with_text, dataset_guidance=guidance,
                               max_text_chars=max_text_chars, max_context_chars=max_context_chars,
                               max_entities=max_entities, max_relations=max_relations)
    return "\n".join(lines) + f"\n\nRECORD:\n{rec}", flags


def truncate_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars <= 0 or len(text or "") <= max_chars:
        return (text or ""), False
    return text[:max_chars].rstrip() + " ...[truncated]", True


def render_record(record: Dict[str, Any], with_text: bool,
                  dataset_guidance: str = "", max_text_chars: int = 0,
                  max_context_chars: int = 0, max_entities: int = 0,
                  max_relations: int = 0) -> Tuple[str, Dict[str, bool]]:
    """Returns (rendered_json_plus_guidance, truncation_flags). Caps default
    to 0 (uncapped, original behaviour) except entities/relations, which
    keep the original hard [:80] safety net regardless -- --max_entities/
    --max_relations tighten that further when set, they don't loosen it.
    """
    flags = {"text": False, "context": False, "entities": False, "relations": False}

    ents = record.get("entities") or []
    ent_cap = max_entities if max_entities > 0 else 80
    flags["entities"] = len(ents) > ent_cap
    ents = ents[:ent_cap]

    rels = record.get("relations") or []
    rel_cap = max_relations if max_relations > 0 else 80
    flags["relations"] = len(rels) > rel_cap
    rels = rels[:rel_cap]

    ctx, flags["context"] = truncate_text(record.get("context_text", ""), max_context_chars)

    payload = {
        "dataset": record.get("dataset"),
        "annotation_level": record.get("annotation_level"),
        "entities": [{"entity_id": str(e.get("entity_id", "")),
                      "type": e.get("type", ""),
                      "text": e.get("text") or (e.get("surface_forms") or [""])[0]}
                     for e in ents],
        "relations": [{"label": r.get("label", ""),
                       "arguments": [{"role": a.get("role", ""),
                                      "entity_id": str(a.get("entity_id", ""))}
                                     for a in (r.get("arguments") or [])]}
                      for r in rels],
        "context": ctx,
    }
    if with_text:
        txt, flags["text"] = truncate_text(record.get("text", ""), max_text_chars)
        payload["text"] = txt
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if dataset_guidance:
        rendered = f"{rendered}\n\nDATASET-SPECIFIC GUIDANCE:\n{dataset_guidance}"
    if any(flags.values()):
        rendered += ("\n\n[NOTE: one or more fields were truncated for length; "
                    "judge only what is visible, do not assume anything about "
                    "truncated content.]")
    return rendered, flags


# =====================================================================
# Backends
# =====================================================================

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
    def __init__(self, model_path: str, system_prompt: str, max_new_tokens: int = 128,
                 max_model_len: int = 8192, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90, dtype: str = "auto",
                 enforce_eager: bool = True, seed: int = 42):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path, dtype=dtype,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, trust_remote_code=True,
                       enforce_eager=enforce_eager, seed=seed)
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.system_prompt = system_prompt

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

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


class EchoBackend:
    """Test double for --self_test."""
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or ['{"hallucination": "not sure"}']
        self.max_model_len = 8192
        self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate_batch(self, prompts: Sequence[str]) -> List[str]:
        return [self.responses[i % len(self.responses)] for i in range(len(prompts))]


# =====================================================================
# Parsing model answers
# =====================================================================

def parse_binary(raw: str) -> Tuple[Optional[str], Optional[str]]:
    obj = extract_json_block(raw)
    if obj is None:
        return None, "json_parse_failure"
    v = obj.get("hallucination")
    if isinstance(v, bool):
        return ("true" if v else "false"), None
    if not isinstance(v, str):
        return None, "missing_hallucination_field"
    v = v.strip().lower()
    if v in {"true", "yes", "hallucinated"}:
        return "true", None
    if v in {"false", "no", "faithful"}:
        return "false", None
    if v in {"not sure", "not_sure", "unsure", "unknown", "uncertain"}:
        return NOT_SURE, None
    return None, f"unrecognised_value:{v[:32]}"


def parse_types(raw: str) -> Tuple[Optional[List[str]], Optional[str]]:
    obj = extract_json_block(raw)
    if obj is None:
        return None, "json_parse_failure"
    v = obj.get("types")
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return None, "missing_types_field"
    out = [str(x).strip() for x in v if str(x).strip() in CATEGORIES]
    if not out:
        return None, "no_valid_category"
    return out, None


# =====================================================================
# Metrics
# =====================================================================

def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def binary_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Binary detection metrics, with abstention made explicit.

    The positive class is 'hallucinated'. Two views are reported:
      decided_only  - 'not sure' and unparseable answers excluded from the
                      denominator (how good the model is when it commits)
      all_instances - abstention counted as an error (deployment view)
    Reporting only one of these is how abstention gets hidden (R2-C8).
    """
    n = len(results)
    n_not_sure = sum(1 for r in results if r["pred"] == NOT_SURE)
    n_unparsed = sum(1 for r in results if r["pred"] is None)
    decided = [r for r in results if r["pred"] in {"true", "false"}]

    tp = sum(1 for r in decided if r["pred"] == "true" and r["gold"])
    fp = sum(1 for r in decided if r["pred"] == "true" and not r["gold"])
    tn = sum(1 for r in decided if r["pred"] == "false" and not r["gold"])
    fn = sum(1 for r in decided if r["pred"] == "false" and r["gold"])

    acc_dec = (tp + tn) / len(decided) if decided else 0.0
    # abstention/parse failure treated as wrong
    acc_all = (tp + tn) / n if n else 0.0
    fn_all = fn + sum(1 for r in results
                      if r["gold"] and r["pred"] in {NOT_SURE, None})
    fp_all = fp
    return {
        "n_instances": n,
        "class_prior_positive": round(sum(1 for r in results if r["gold"]) / n, 4) if n else 0.0,
        "not_sure_rate": round(n_not_sure / n, 4) if n else 0.0,
        "unparsed_rate": round(n_unparsed / n, 4) if n else 0.0,
        "decided_coverage": round(len(decided) / n, 4) if n else 0.0,
        "confusion_matrix_decided": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "decided_only": {"accuracy": round(acc_dec, 4), **prf(tp, fp, fn)},
        "all_instances": {"accuracy": round(acc_all, 4), **prf(tp, fp_all, fn_all),
                          "note": "abstention and parse failures counted as errors"},
    }


def type_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Type-classification metrics + full confusion matrix (R1-C2)."""
    decided = [r for r in results if r["pred_types"]]
    n = len(results)
    top1 = sum(1 for r in decided if r["pred_types"][0] == r["gold_category"])
    exact = sum(1 for r in decided if set(r["pred_types"]) == {r["gold_category"]})

    per_cat: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
    confusion: Dict[str, Counter] = {c: Counter() for c in CATEGORIES}
    for r in decided:
        gold = r["gold_category"]
        pred = r["pred_types"][0]
        confusion[gold][pred] += 1
        for c in CATEGORIES:
            in_pred = c in r["pred_types"]
            in_gold = (c == gold)
            if in_pred and in_gold:
                per_cat[c]["tp"] += 1
            elif in_pred and not in_gold:
                per_cat[c]["fp"] += 1
            elif in_gold and not in_pred:
                per_cat[c]["fn"] += 1

    per_cat_scores = {c: {**v, **prf(v["tp"], v["fp"], v["fn"])} for c, v in per_cat.items()}
    macro_f1 = round(sum(v["f1"] for v in per_cat_scores.values()) / len(CATEGORIES), 4)
    micro = prf(sum(v["tp"] for v in per_cat.values()),
                sum(v["fp"] for v in per_cat.values()),
                sum(v["fn"] for v in per_cat.values()))
    return {
        "n_instances": n,
        "decided_coverage": round(len(decided) / n, 4) if n else 0.0,
        "top1_accuracy": round(top1 / len(decided), 4) if decided else 0.0,
        "exact_set_match": round(exact / len(decided), 4) if decided else 0.0,
        "macro_f1": macro_f1,
        "micro": micro,
        "per_category": per_cat_scores,
        "confusion_matrix": {g: dict(c) for g, c in confusion.items()},
        "note": "rows = gold category, columns = predicted top-1",
    }


def detection_baselines(results: Sequence[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    """Trivial responders on the SAME evaluation set (R1-C4).

    If always-yes beats the models, the reported numbers are not evidence of
    detection ability. This must appear alongside the model rows.
    """
    rng = random.Random(seed)
    n = len(results)
    if not n:
        return {}
    gold = [r["gold"] for r in results]
    prior = sum(gold) / n

    def score(preds):
        tp = sum(1 for p, g in zip(preds, gold) if p and g)
        fp = sum(1 for p, g in zip(preds, gold) if p and not g)
        fn = sum(1 for p, g in zip(preds, gold) if not p and g)
        tn = sum(1 for p, g in zip(preds, gold) if not p and not g)
        return {"accuracy": round((tp + tn) / n, 4), **prf(tp, fp, fn)}

    return {
        "class_prior_positive": round(prior, 4),
        "always_yes": score([True] * n),
        "always_no": score([False] * n),
        "random_uniform": score([rng.random() < 0.5 for _ in range(n)]),
        "random_stratified": score([rng.random() < prior for _ in range(n)]),
    }


def bootstrap_binary_ci(results: Sequence[Dict[str, Any]], n_boot: int = 1000,
                        alpha: float = 0.05, seed: int = 42) -> Dict[str, Any]:
    """Percentile CI clustered by SOURCE ROW.

    Instances derived from the same source row share text and most of their
    relations, so they are not independent. Resampling instances instead of
    source rows would give intervals several times too narrow.
    """
    if not results:
        return {}
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_src[r["source_id"]].append(r)
    clusters = list(by_src.values())
    rng = random.Random(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        sample: List[Dict[str, Any]] = []
        for _ in range(len(clusters)):
            sample.extend(clusters[rng.randrange(len(clusters))])
        m = binary_metrics(sample)
        accs.append(m["decided_only"]["accuracy"])
        f1s.append(m["decided_only"]["f1"])
    def ci(v):
        v = sorted(v)
        return {"lo": round(v[int((alpha / 2) * len(v))], 4),
                "hi": round(v[min(len(v) - 1, int((1 - alpha / 2) * len(v)))], 4)}
    return {"n_boot": n_boot, "level": 1 - alpha, "n_clusters": len(clusters),
            "accuracy_ci": ci(accs), "f1_ci": ci(f1s)}


def stratify(results: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[str(r.get(key))].append(r)
    return {g: binary_metrics(v) for g, v in sorted(groups.items())}


def corruption_bucket(cr: Optional[float]) -> str:
    if cr is None:
        return "unknown"
    if cr <= 0.0:
        return "0.0 (negative)"
    if cr < 0.15:
        return "<0.15"
    if cr < 0.35:
        return "0.15-0.34"
    if cr < 0.6:
        return "0.35-0.59"
    return "0.6+"


# =====================================================================
# Driver
# =====================================================================

def run(args: argparse.Namespace) -> int:
    setup_logging(args.output_dir, args.log_level)
    rng = random.Random(args.seed)

    hallu = list(load_jsonl(args.hallu_jsonl))
    unified = list(load_jsonl(args.unified_jsonl))
    LOGGER.info("Loaded %d hallucination rows, %d unified rows.", len(hallu), len(unified))

    instances = build_eval_set(hallu, unified, args.negative_ratio,
                               args.seed, args.max_per_cell)
    LOGGER.info("Evaluation set: %d instances (%d positive, %d negative)",
                len(instances),
                sum(1 for i in instances if i.is_hallucinated),
                sum(1 for i in instances if not i.is_hallucinated))

    bundle = load_prompt_bundle(args.prompt_dir)
    LOGGER.info("Prompt source: %s%s", bundle.source,
               f" (dataset guidance loaded for: {sorted(bundle.dataset_guidance)})"
               if bundle.dataset_guidance else "")

    if args.backend == "vllm":
        backend = VLLMBackend(args.model_path, bundle.system_prompt,
                              max_new_tokens=args.max_new_tokens,
                              max_model_len=args.max_model_len,
                              tensor_parallel_size=args.tensor_parallel_size,
                              gpu_memory_utilization=args.gpu_memory_utilization,
                              dtype=args.dtype, enforce_eager=args.enforce_eager,
                              seed=args.seed)
    else:
        backend = EchoBackend()

    out: Dict[str, Any] = {"config": vars(args).copy()}
    out["config"]["prompt_source"] = bundle.source
    records_out: List[Dict[str, Any]] = []

    def log_truncation(all_flags: List[Dict[str, bool]], task_name: str) -> None:
        counts = Counter()
        for f in all_flags:
            for field, hit in f.items():
                if hit:
                    counts[field] += 1
        if sum(counts.values()):
            LOGGER.warning("%s: truncation applied to keep prompts under the context "
                          "window: %s (out of %d instances). Increase --max_text_chars/"
                          "--max_entities/--max_relations/--max_context_chars, or "
                          "--max_model_len, if you want this at 0.",
                          task_name, dict(counts), len(all_flags))

    # ---------------- binary ----------------
    if args.task in {"binary", "both"}:
        prompts_flags = [build_binary_prompt(i, args.with_text, bundle,
                                            args.max_text_chars, args.max_context_chars,
                                            args.max_entities, args.max_relations)
                         for i in instances]
        prompts = [p for p, _ in prompts_flags]
        log_truncation([f for _, f in prompts_flags], "binary")
        raws: List[str] = []
        for s in range(0, len(prompts), args.batch_size):
            raws.extend(backend.generate_batch(prompts[s:s + args.batch_size]))
            LOGGER.info("binary %d/%d", min(s + args.batch_size, len(prompts)), len(prompts))

        results = []
        for inst, raw in zip(instances, raws):
            pred, err = parse_binary(raw)
            results.append({"instance_id": inst.instance_id, "source_id": inst.source_id,
                            "dataset": inst.dataset, "gold": inst.is_hallucinated,
                            "gold_category": inst.category, "pred": pred,
                            "parse_error": err, "raw": raw,
                            "corruption_bucket": corruption_bucket(inst.corruption_rate)})
        records_out.extend({**r, "task": "binary"} for r in results)

        out["binary"] = {
            "overall": binary_metrics(results),
            "baselines": detection_baselines(results, seed=args.seed),
            "bootstrap_ci": bootstrap_binary_ci(results, n_boot=args.bootstrap,
                                                seed=args.seed) if args.bootstrap else {},
            "by_dataset": stratify(results, "dataset"),
            "by_gold_category": stratify(results, "gold_category"),
            "by_corruption_rate": stratify(results, "corruption_bucket"),
            "abstention": {
                "overall_not_sure_rate": binary_metrics(results)["not_sure_rate"],
                "by_dataset": {d: m["not_sure_rate"]
                               for d, m in stratify(results, "dataset").items()},
                "by_gold_category": {c: m["not_sure_rate"]
                                     for c, m in stratify(results, "gold_category").items()},
            },
            "parse_errors": dict(Counter(r["parse_error"] for r in results
                                         if r["parse_error"])),
        }

    # ---------------- type classification ----------------
    if args.task in {"type", "both"}:
        pos = [i for i in instances if i.is_hallucinated]
        prompts_flags = [build_type_prompt(i, args.with_text, args.prompt_variant, rng, bundle,
                                          args.max_text_chars, args.max_context_chars,
                                          args.max_entities, args.max_relations)
                         for i in pos]
        prompts = [p for p, _ in prompts_flags]
        log_truncation([f for _, f in prompts_flags], "type")
        raws = []
        for s in range(0, len(prompts), args.batch_size):
            raws.extend(backend.generate_batch(prompts[s:s + args.batch_size]))
            LOGGER.info("type %d/%d", min(s + args.batch_size, len(prompts)), len(prompts))

        tresults = []
        for inst, raw in zip(pos, raws):
            types, err = parse_types(raw)
            tresults.append({"instance_id": inst.instance_id, "source_id": inst.source_id,
                             "dataset": inst.dataset, "gold_category": inst.category,
                             "pred_types": types, "parse_error": err, "raw": raw,
                             "corruption_bucket": corruption_bucket(inst.corruption_rate)})
        records_out.extend({**r, "task": "type"} for r in tresults)

        by_ds = defaultdict(list)
        for r in tresults:
            by_ds[r["dataset"]].append(r)
        out["type_classification"] = {
            "prompt_variant": args.prompt_variant,
            "overall": type_metrics(tresults),
            "by_dataset": {d: type_metrics(v) for d, v in sorted(by_ds.items())},
            "parse_errors": dict(Counter(r["parse_error"] for r in tresults
                                         if r["parse_error"])),
            "note": ("prompt_variant=neutral randomises category order per instance and "
                     "contains no priority ordering; 'priority' reproduces the previous "
                     "prompt's hard-coded prior and exists only as an ablation."),
        }

    save_json(out, os.path.join(args.output_dir, "detection_summary.json"))
    if args.save_predictions:
        save_jsonl(records_out, os.path.join(args.output_dir, "detection_predictions.jsonl"))

    # ---------------- console report ----------------
    if "binary" in out:
        b = out["binary"]["overall"]
        bl = out["binary"]["baselines"]
        LOGGER.info("=" * 72)
        LOGGER.info("BINARY DETECTION")
        LOGGER.info("  instances=%d  positive prior=%.3f", b["n_instances"],
                    b["class_prior_positive"])
        LOGGER.info("  not-sure rate=%.3f  decided coverage=%.3f  unparsed=%.3f",
                    b["not_sure_rate"], b["decided_coverage"], b["unparsed_rate"])
        LOGGER.info("  decided-only : acc=%.3f  F1=%.3f",
                    b["decided_only"]["accuracy"], b["decided_only"]["f1"])
        LOGGER.info("  all-instances: acc=%.3f  F1=%.3f",
                    b["all_instances"]["accuracy"], b["all_instances"]["f1"])
        LOGGER.info("  BASELINE always-yes: acc=%.3f F1=%.3f  <- must be beaten",
                    bl["always_yes"]["accuracy"], bl["always_yes"]["f1"])
    if "type_classification" in out:
        t = out["type_classification"]["overall"]
        LOGGER.info("TYPE CLASSIFICATION (variant=%s)", args.prompt_variant)
        LOGGER.info("  top1=%.3f  exact-set=%.3f  macro-F1=%.3f",
                    t["top1_accuracy"], t["exact_set_match"], t["macro_f1"])
    LOGGER.info("=" * 72)
    LOGGER.info("Summary: %s", os.path.join(args.output_dir, "detection_summary.json"))
    return 0


# =====================================================================
# Self-test
# =====================================================================

def self_test() -> int:
    fails = 0

    print("=== answer parsing ===")
    for raw, want in [('{"hallucination": "true"}', "true"),
                      ('{"hallucination": "false"}', "false"),
                      ('{"hallucination": "not sure"}', NOT_SURE),
                      ('<think>hmm {</think>{"hallucination":"not_sure"}', NOT_SURE),
                      ('{"hallucination": true}', "true"),
                      ('garbage', None)]:
        got, err = parse_binary(raw)
        ok = got == want
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {raw[:44]:46s} -> {got}")

    print("\n=== abstention is visible in metrics ===")
    res = ([{"pred": "true", "gold": True, "source_id": f"s{i}"} for i in range(40)] +
           [{"pred": NOT_SURE, "gold": True, "source_id": f"s{i}"} for i in range(40, 70)] +
           [{"pred": "false", "gold": False, "source_id": f"s{i}"} for i in range(70, 100)])
    m = binary_metrics(res)
    ok = (m["not_sure_rate"] == 0.3 and m["decided_coverage"] == 0.7
          and m["decided_only"]["accuracy"] == 1.0 and m["all_instances"]["accuracy"] == 0.7)
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] not_sure={m['not_sure_rate']} "
          f"coverage={m['decided_coverage']} "
          f"acc_decided={m['decided_only']['accuracy']} "
          f"acc_all={m['all_instances']['accuracy']}")
    print("       (decided-only hides the 30% abstention; all-instances exposes it)")

    print("\n=== always-yes baseline on an 80/20 split ===")
    res2 = ([{"pred": "true", "gold": True, "source_id": f"s{i}"} for i in range(80)] +
            [{"pred": "true", "gold": False, "source_id": f"s{i}"} for i in range(80, 100)])
    bl = detection_baselines(res2)
    ok = bl["always_yes"]["f1"] > 0.85
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] prior={bl['class_prior_positive']} "
          f"always_yes F1={bl['always_yes']['f1']} acc={bl['always_yes']['accuracy']}")

    print("\n=== type metrics + confusion matrix ===")
    tr = [{"gold_category": "relation_hallucination", "pred_types": ["relation_hallucination"],
           "source_id": "a", "dataset": "X"},
          {"gold_category": "incompleteness", "pred_types": ["relation_hallucination"],
           "source_id": "b", "dataset": "X"},
          {"gold_category": "overgeneration", "pred_types": ["overgeneration"],
           "source_id": "c", "dataset": "X"}]
    tm = type_metrics(tr)
    ok = (tm["top1_accuracy"] == round(2 / 3, 4)
          and tm["confusion_matrix"]["incompleteness"]["relation_hallucination"] == 1)
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] top1={tm['top1_accuracy']} macro_f1={tm['macro_f1']}")
    print(f"       confusion[incompleteness] = {tm['confusion_matrix']['incompleteness']}")

    print("\n=== prompt variants ===")
    test_bundle = PromptBundle(
        system_prompt="You are a test evaluator.",
        binary_instruction='TASK: decide. Return ONLY JSON: {"hallucination": "true"}',
        type_head="TASK: identify which kind.",
        type_tail='Return ONLY JSON: {"types": ["category_name"]}',
        category_gloss={
            "relation_hallucination": "wrong label",
            "incompleteness": "missing relation",
            "overgeneration": "extra relation",
            "context_induced": "context-induced relation",
        },
        dataset_guidance={},
        source="self_test fixture",
    )
    inst = Instance("i", "s", "DDI", {"dataset": "DDI", "relations": [], "entities": []},
                    True, "incompleteness", 0.5, 2)
    neutral, _ = build_type_prompt(inst, True, "neutral", random.Random(0), test_bundle)
    priority, _ = build_type_prompt(inst, True, "priority", random.Random(0), test_bundle)
    ok = ("Strict priority" not in neutral) and ("Strict priority" in priority)
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] neutral has no priority list; ablation does "
          f"(still code-only regardless of --prompt_dir)")

    print("\n=== answer leakage ===")
    row = {"benchmark_id": "x", "hallucination_category": "incompleteness",
           "hallucination_provenance": {"flips": []}, "metadata": {"relation_label_inventory": {}},
           "is_hallucination_benchmark": True, "source_benchmark_id": "y",
           "relations": [], "entities": [], "text": "t"}
    clean = strip_prompt_unsafe(row)
    leaked = [k for k in PROMPT_UNSAFE_KEYS if k in clean]
    ok = not leaked
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] stripped; remaining unsafe keys: {leaked}")

    print("\n=== --prompt_dir is required, with no built-in fallback ===")
    import tempfile

    def make_system_txt(sys_text="Judge each category on its own evidence.",
                        binary_text='TASK: decide. Return JSON: {"hallucination": "true"}',
                        head_text="TASK: identify which kind.",
                        tail_text='Return JSON: {"types": ["category_name"]}',
                        cats=None) -> str:
        cats = cats or {"relation_hallucination": "wrong label",
                        "incompleteness": "missing relation",
                        "overgeneration": "extra relation",
                        "context_induced": "context-induced relation"}
        return (f"### SYSTEM ###\n{sys_text}\n\n"
               f"### BINARY_INSTRUCTION ###\n{binary_text}\n\n"
               f"### TYPE_INSTRUCTION_HEAD ###\n{head_text}\n\n"
               f"### TYPE_INSTRUCTION_TAIL ###\n{tail_text}\n\n"
               f"### CATEGORIES ###\n{json.dumps(cats)}\n")

    ok = False
    try:
        load_prompt_bundle("")
    except ValueError:
        ok = True
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] empty --prompt_dir refuses (no built-in prompts exist)")

    with tempfile.TemporaryDirectory() as td:
        empty_dir = Path(td) / "empty"
        empty_dir.mkdir()
        ok = False
        try:
            load_prompt_bundle(str(empty_dir))
        except ValueError as e:
            ok = "missing required file" in str(e)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] a directory missing system.txt refuses "
              f"with a specific message, not a crash")

        complete_dir = Path(td) / "complete"
        complete_dir.mkdir()
        (complete_dir / "system.txt").write_text(make_system_txt(), encoding="utf-8")
        (complete_dir / "DDI.txt").write_text("Valid labels: MECHANISM, EFFECT, ADVISE, INT.",
                                               encoding="utf-8")
        loaded = load_prompt_bundle(str(complete_dir))
        ok = (loaded.source == str(complete_dir)
              and "DDI" in loaded.dataset_guidance
              and "CHEMPROT" not in loaded.dataset_guidance
              and loaded.binary_instruction.startswith("TASK: decide"))
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] one sectioned system.txt loads all five "
              f"pieces correctly; missing optional dataset files fall back without crashing")

        prompt, _ = build_binary_prompt(inst, True, loaded)
        ok = "MECHANISM" in prompt
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] loaded dataset guidance actually reaches "
              f"the built prompt")

        # bias inside any section is caught the same way as before
        (complete_dir / "system.txt").write_text(
            make_system_txt(sys_text="Strict priority:\n1. relation_hallucination "
                                    "(strongest signal)"), encoding="utf-8")
        ok = False
        try:
            load_prompt_bundle(str(complete_dir))
        except ValueError:
            ok = True
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] a section containing the documented "
              f"bias pattern still refuses to load")

        # CATEGORIES with the wrong key set is caught explicitly
        (complete_dir / "system.txt").write_text(
            make_system_txt(cats={"relation_hallucination": "x"}), encoding="utf-8")
        ok = False
        try:
            load_prompt_bundle(str(complete_dir))
        except ValueError as e:
            ok = "must define exactly these categories" in str(e)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] CATEGORIES missing a category is "
              f"caught with a specific error")

        # a missing section is caught explicitly, not silently defaulted
        bad_text = make_system_txt().replace("### CATEGORIES ###", "### CATEGORYY ###")
        (complete_dir / "system.txt").write_text(bad_text, encoding="utf-8")
        ok = False
        try:
            load_prompt_bundle(str(complete_dir))
        except ValueError as e:
            ok = "missing required section" in str(e)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] a typo'd section marker is caught "
              f"as a missing section, not silently dropped")

        # an unrecognised section is caught too, not silently ignored
        stray = make_system_txt() + "\n### EXTRA_SECTION ###\nwhoops\n"
        (complete_dir / "system.txt").write_text(stray, encoding="utf-8")
        ok = False
        try:
            load_prompt_bundle(str(complete_dir))
        except ValueError as e:
            ok = "unrecognised section" in str(e)
        fails += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] an unrecognised section is flagged, "
              f"not silently ignored")

    import inspect
    ok = "system_prompt" in inspect.signature(VLLMBackend.__init__).parameters
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] VLLMBackend requires an external system_prompt "
          f"(no module-level default to fall back on)")

    print("\n=== truncation: the actual bug that broke Med42-8B in prod ===")
    huge_row = {"dataset": "DDI",
               "text": "x" * 20000, "context_text": "y" * 10000,
               "entities": [{"entity_id": f"e{i}", "type": "DRUG", "text": f"drug{i}"}
                           for i in range(80)],
               "relations": [{"label": "EFFECT", "arguments": [
                   {"role": "e1", "entity_id": f"e{i}"}, {"role": "e2", "entity_id": f"e{i+1}"}]}
                             for i in range(80)]}
    huge_inst = Instance("h", "s", "DDI", huge_row, True, "relation_hallucination", 0.5, 80)

    no_cap, flags_no_cap = build_binary_prompt(huge_inst, True, test_bundle)
    ok = len(no_cap) > 20000 and not any(flags_no_cap.values())
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] no caps (0 = uncapped, default) -- this is what "
          f"just broke Med42-8B in production", f"len={len(no_cap)} flags={flags_no_cap}")

    capped, flags_capped = build_binary_prompt(huge_inst, True, test_bundle,
                                               max_text_chars=2000, max_context_chars=1000,
                                               max_entities=20, max_relations=20)
    ok = len(capped) < len(no_cap) and flags_capped["text"] and flags_capped["entities"]
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] capped prompt is dramatically smaller and flags "
          f"correctly report what was cut", f"capped={len(capped)} vs uncapped={len(no_cap)}")

    ok = "[NOTE: one or more fields were truncated" in capped
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] truncated prompt tells the model it was truncated")

    small_row = {"dataset": "DDI", "text": "short", "context_text": "",
                "entities": [{"entity_id": "e0", "type": "DRUG", "text": "d"}],
                "relations": [{"label": "EFFECT", "arguments": [
                    {"role": "e1", "entity_id": "e0"}, {"role": "e2", "entity_id": "e0"}]}]}
    small_inst = Instance("sm", "s", "DDI", small_row, True, "relation_hallucination", 0.5, 1)
    _, flags_small = build_binary_prompt(small_inst, True, test_bundle,
                                        max_text_chars=2000, max_context_chars=1000,
                                        max_entities=20, max_relations=20)
    ok = not any(flags_small.values())
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] a record well under the cap is never marked truncated")

    # zero-cap default still applies the original fixed 80-item safety net
    huge_ents_row = dict(huge_row)
    huge_ents_row["entities"] = [{"entity_id": f"e{i}", "type": "DRUG", "text": f"d{i}"}
                                 for i in range(200)]
    huge_ents_inst = Instance("he", "s", "DDI", huge_ents_row, True,
                              "relation_hallucination", 0.5, 80)
    _, flags_default = build_binary_prompt(huge_ents_inst, True, test_bundle)
    ok = flags_default["entities"]
    fails += not ok
    print(f"  [{'PASS' if ok else 'FAIL'}] even with all caps at 0, the original fixed "
          f"80-entity safety net still applies")

    total = 25
    print(f"\n{total - fails}/{total} checks passed.")
    return 1 if fails else 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HalluRE-Med hallucination detection evaluation.")
    p.add_argument("--hallu_jsonl", type=str, default="")
    p.add_argument("--unified_jsonl", type=str, default="")
    p.add_argument("--output_dir", type=str, default="outputs/detection")
    p.add_argument("--model_path", type=str, default="")

    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "echo"])
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_text_chars", type=int, default=0,
                   help="Cap on the TEXT field before rendering. 0 = no cap (original "
                        "behaviour). Real corpus measurement: the worst-case prompt "
                        "(DDI_DDI-DrugBank.d64.s87, 55 entities) needs ~8586 tokens "
                        "uncapped -- set caps here if --max_model_len is at or below "
                        "your model's own context ceiling and can't just be raised.")
    p.add_argument("--max_context_chars", type=int, default=0, help="0 = no cap.")
    p.add_argument("--max_entities", type=int, default=0,
                   help="0 = fall back to the fixed 80-entity safety cap already applied "
                        "regardless. Set lower to fit a smaller --max_model_len.")
    p.add_argument("--max_relations", type=int, default=0,
                   help="0 = fall back to the fixed 80-relation safety cap already "
                        "applied regardless. Set lower to fit a smaller --max_model_len.")

    p.add_argument("--task", type=str, default="both", choices=["binary", "type", "both"])
    p.add_argument("--with_text", action="store_true", default=True,
                   help="Include the source TEXT in the record shown to the model.")
    p.add_argument("--without_text", dest="with_text", action="store_false")
    p.add_argument("--prompt_variant", type=str, default="neutral",
                   choices=["neutral", "priority"],
                   help="neutral = randomised category order, no priority list (primary). "
                        "priority = reproduces the old prompt's hard-coded prior (ablation).")
    p.add_argument("--prompt_dir", type=str, default="prompts_hallucination_detection",
                   help="Directory containing system.txt (required -- five "
                        "'### NAME ###'-delimited sections: SYSTEM, BINARY_INSTRUCTION, "
                        "TYPE_INSTRUCTION_HEAD, TYPE_INSTRUCTION_TAIL, CATEGORIES), and "
                        "optionally CHEMPROT.txt/DDI.txt/DCE.txt/BIORED.txt for extra "
                        "per-dataset grounding. This script has no built-in prompt text. "
                        "Every section is scanned for the documented hard-coded-priority "
                        "bias and the run refuses to start if found.")
    p.add_argument("--negative_ratio", type=float, default=1.0,
                   help="Non-hallucinated instances per hallucinated instance, per dataset.")
    p.add_argument("--max_per_cell", type=int, default=None,
                   help="Cap instances per (dataset, category) cell.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--save_predictions", action="store_true", default=True)
    p.add_argument("--no_save_predictions", dest="save_predictions", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--self_test", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.self_test:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return self_test()
    if not args.hallu_jsonl or not args.unified_jsonl:
        print("--hallu_jsonl and --unified_jsonl are required (or --self_test).",
              file=sys.stderr)
        return 2
    if args.backend == "vllm" and not args.model_path:
        print("--model_path is required for --backend vllm.", file=sys.stderr)
        return 2
    try:
        return run(args)
    except ValueError as exc:
        # load_prompt_bundle raises this for a missing/incomplete --prompt_dir
        # or a bias-pattern refusal -- surface it as a clean CLI error rather
        # than a raw traceback.
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())