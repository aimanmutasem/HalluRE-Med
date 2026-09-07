#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("re_eval")

ALL_DATASETS = ["CHEMPROT", "DDI", "DCE", "BIORED"]
DATASET_LABELS = {
    "CHEMPROT": {"CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"},
    "DDI": {"MECHANISM", "EFFECT", "ADVISE", "INT"},
    "DCE": {"POS", "COMB", "NEG"},
    "BIORED": {"Association", "Positive_Correlation", "Negative_Correlation", "Bind",
               "Comparison", "Drug_Interaction", "Cotreatment", "Conversion"},
}
BIORED_ALLOWED_TYPES = {"GENE", "DISEASE", "CHEMICAL", "VARIANT"}
DEFAULT_FUZZY_MIN_SCORE = 60.0


# ============================ text helpers ============================

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_match_text(text: str) -> str:
    text = normalize_space(text).lower()
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"^[\s\.,;:()\[\]{}\"'`]+", "", text)
    text = re.sub(r"[\s\.,;:()\[\]{}\"'`]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_reasoning(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks before JSON extraction."""
    if "<think>" not in text:
        return text
    while "<think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            return text[:start]
        text = text[:start] + text[end + len("</think>"):]
    return text.strip()


def extract_json_block(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not raw or not raw.strip():
        return None, "empty_output"
    text = strip_reasoning(raw.strip()).strip()
    if not text:
        return None, "reasoning_only_no_answer"
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    if start == -1:
        return None, "no_json_object"
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
                    return None, "json_decode_error"
                return (obj, None) if isinstance(obj, dict) else (None, "not_an_object")
    return None, "unbalanced_json_likely_truncated"


def parse_prediction(raw: str) -> Tuple[Dict[str, Any], Optional[str]]:
    obj, err = extract_json_block(raw)
    if obj is None:
        return {"relations": []}, err
    rels = obj.get("relations")
    if not isinstance(rels, list):
        return {"relations": []}, "relations_not_list"
    return {"relations": rels}, None


# ========================= type normalization =========================

def normalize_type_general(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    tl = t.lower()
    if tl in {"chemical", "chem", "chemicalentity"}:
        return "CHEMICAL"
    if tl in {"gene", "protein", "receptor", "gene/protein", "gene-protein",
              "geneorgeneproduct"}:
        return "GENE"
    if tl in {"drug"}:
        return "DRUG"
    if tl in {"disease", "diseaseorphenotypicfeature"}:
        return "DISEASE"
    if tl in {"variant", "sequencevariant"}:
        return "VARIANT"
    return t.upper()


def normalize_type_for_dataset(dataset: str, t: str) -> str:
    base = normalize_type_general(t)
    if dataset == "CHEMPROT":
        if base == "CHEMICAL":
            return "CHEMICAL"
        if base in {"GENE", "GENE-Y", "GENE-N"}:
            return "GENE"
    if dataset == "DDI" and base in {"DRUG", "CHEMICAL", "BRAND", "GROUP", "DRUG_N"}:
        return "DRUG"
    if dataset == "DCE":
        return "DRUG"
    if dataset == "BIORED":
        return base if base in BIORED_ALLOWED_TYPES else "OTHER"
    return base


def is_cp_chem(t: str) -> bool:
    return normalize_type_for_dataset("CHEMPROT", t) == "CHEMICAL"


def is_cp_gene(t: str) -> bool:
    return normalize_type_for_dataset("CHEMPROT", t) == "GENE"


# ========================= canonicalization ===========================

def canonical_pair_rel(rel: Dict[str, Any], sort_pair: bool = False) -> Tuple:
    pair = [(normalize_match_text(rel.get("entity1", "")), rel.get("entity1_type", "")),
            (normalize_match_text(rel.get("entity2", "")), rel.get("entity2_type", ""))]
    if sort_pair:
        pair = sorted(pair)
    return (tuple(pair), rel.get("relation", ""))


def canonical_dce_rel(rel: Dict[str, Any]) -> Tuple:
    ents = rel.get("entities", []) or []
    return (tuple(sorted((normalize_match_text(x.get("text", "")), x.get("type", ""))
                         for x in ents)), rel.get("relation", ""))


def chemprot_reorder(rel: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    e1, e2 = normalize_space(rel.get("entity1", "")), normalize_space(rel.get("entity2", ""))
    t1 = normalize_type_for_dataset("CHEMPROT", rel.get("entity1_type", ""))
    t2 = normalize_type_for_dataset("CHEMPROT", rel.get("entity2_type", ""))
    label = rel.get("relation", "")
    if not e1 or not e2 or not label:
        return None, "missing_field"
    if is_cp_chem(t1) and is_cp_gene(t2):
        return {"entity1": e1, "entity1_type": "CHEMICAL",
                "entity2": e2, "entity2_type": "GENE", "relation": label}, None
    if is_cp_gene(t1) and is_cp_chem(t2):
        return {"entity1": e2, "entity1_type": "CHEMICAL",
                "entity2": e1, "entity2_type": "GENE", "relation": label}, None
    return None, "invalid_type_pairing"


# =========================== gold extraction ==========================

def get_entity_text_by_id(row: Dict[str, Any], eid: str) -> Tuple[str, str]:
    for e in row.get("entities", []) or []:
        if str(e.get("entity_id")) == str(eid):
            if e.get("text"):
                return e["text"], e.get("type", "")
            sfs = e.get("surface_forms", []) or []
            return (sfs[0] if sfs else str(eid)), e.get("type", "")
    return str(eid), ""


def gold_relations_pairwise(row: Dict[str, Any], include_non: bool = False) -> List[Dict[str, Any]]:
    ds = row.get("dataset", "")
    out = []
    for rel in row.get("relations", []) or []:
        label = rel.get("label", "")
        if ds == "DDI" and label == "NON" and not include_non:
            continue
        args = rel.get("arguments", []) or []
        if len(args) != 2:
            continue
        t1, ty1 = get_entity_text_by_id(row, args[0].get("entity_id"))
        t2, ty2 = get_entity_text_by_id(row, args[1].get("entity_id"))
        out.append({"entity1": t1, "entity1_type": normalize_type_for_dataset(ds, ty1),
                    "entity2": t2, "entity2_type": normalize_type_for_dataset(ds, ty2),
                    "relation": label})
    return out


def gold_relations_dce(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for rel in row.get("relations", []) or []:
        ents = [{"text": get_entity_text_by_id(row, a.get("entity_id"))[0], "type": "DRUG"}
                for a in (rel.get("arguments", []) or [])]
        if len(ents) >= 2:
            out.append({"entities": ents, "relation": rel.get("label", "")})
    return out


# ====================== BioRED entity-level matching ==================

def build_biored_index(row: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    index: Dict[str, List[Tuple[str, str]]] = {}
    for e in row.get("entities", []) or []:
        etype = normalize_type_for_dataset("BIORED", e.get("type", ""))
        if etype not in BIORED_ALLOWED_TYPES:
            continue
        forms = list(e.get("surface_forms", []) or [])
        if e.get("text"):
            forms.append(e["text"])
        norm = [(normalize_match_text(f), etype) for f in forms if f]
        if norm:
            index[str(e.get("entity_id"))] = norm
    return index


def fuzzy_score(pred: str, alias: str) -> float:
    if not pred or not alias:
        return 0.0
    if pred == alias:
        return 100.0
    p_tok, a_tok = set(pred.split()), set(alias.split())
    if p_tok and p_tok == a_tok:
        return 95.0
    if len(pred) >= 4 and len(alias) >= 4 and (pred in alias or alias in pred):
        lo, hi = sorted((len(pred), len(alias)))
        return 80.0 * lo / hi
    if p_tok and a_tok:
        jacc = len(p_tok & a_tok) / len(p_tok | a_tok)
        if jacc >= 0.5:
            return 60.0 + 20.0 * jacc
    return 0.0


def match_biored_entity(text: str, ptype: str, index, min_score: float) -> Optional[str]:
    pred_norm = normalize_match_text(text)
    ptype = normalize_type_for_dataset("BIORED", ptype)
    best_id, best_score, best_len = None, 0.0, 10 ** 9
    for eid, aliases in index.items():
        for alias, etype in aliases:
            if ptype in BIORED_ALLOWED_TYPES and etype != ptype:
                continue
            s = fuzzy_score(pred_norm, alias)
            if s > best_score or (s == best_score and s > 0 and len(alias) < best_len):
                best_id, best_score, best_len = eid, s, len(alias)
    return best_id if best_score >= min_score else None


def gold_biored_pairs(row: Dict[str, Any]) -> set:
    out = set()
    for rel in row.get("relations", []) or []:
        args = rel.get("arguments", []) or []
        if len(args) != 2:
            continue
        ids = tuple(sorted(str(a.get("entity_id")) for a in args))
        out.add((ids, rel.get("label", "")))
    return out


# ================== prediction normalization (both modes) =============

@dataclass
class NormalizedPreds:
    kept: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[Tuple[Dict[str, Any], str]] = field(default_factory=list)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


def normalize_predictions(dataset: str, rels: Sequence[Any]) -> NormalizedPreds:
    """Split predictions into schema-valid (`kept`) and schema-invalid (`dropped`).

    LENIENT scoring uses `kept` only (the previous pipeline's behaviour).
    STRICT scoring additionally counts `dropped` as false positives.
    """
    out = NormalizedPreds()
    labels = DATASET_LABELS.get(dataset, set())
    for rel in rels:
        if not isinstance(rel, dict):
            out.dropped.append(({"raw": str(rel)[:200]}, "not_an_object"))
            continue
        if dataset == "DCE":
            ents, label = rel.get("entities"), rel.get("relation", "")
            if not isinstance(ents, list) or len(ents) < 2:
                out.dropped.append((rel, "dce_too_few_entities")); continue
            if labels and label not in labels:
                out.dropped.append((rel, "illegal_label")); continue
            out.kept.append({"entities": [{"text": normalize_space(e.get("text", "")),
                                           "type": "DRUG"}
                                          for e in ents if isinstance(e, dict)],
                             "relation": label})
            continue
        label = rel.get("relation", "")
        if labels and label not in labels:
            out.dropped.append((rel, "illegal_label")); continue
        if dataset == "DDI" and label == "NON":
            out.dropped.append((rel, "predicted_NON")); continue
        if dataset == "CHEMPROT":
            fixed, why = chemprot_reorder(rel)
            if fixed is None:
                out.dropped.append((rel, why or "invalid")); continue
            out.kept.append(fixed); continue
        e1, e2 = normalize_space(rel.get("entity1", "")), normalize_space(rel.get("entity2", ""))
        if not e1 or not e2:
            out.dropped.append((rel, "missing_entity")); continue
        t1 = normalize_type_for_dataset(dataset, rel.get("entity1_type", ""))
        t2 = normalize_type_for_dataset(dataset, rel.get("entity2_type", ""))
        if dataset == "BIORED" and (t1 not in BIORED_ALLOWED_TYPES
                                    or t2 not in BIORED_ALLOWED_TYPES):
            out.dropped.append((rel, "biored_disallowed_type")); continue
        out.kept.append({"entity1": e1, "entity1_type": t1,
                         "entity2": e2, "entity2_type": t2, "relation": label})
    return out


# ============================ scoring =================================

@dataclass
class RowScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold_n: int = 0
    pred_n: int = 0
    dropped_n: int = 0
    mapping_failures: int = 0


def score_row(row, preds: NormalizedPreds, *, strict: bool, ddi_directional: bool,
              fuzzy_min: float, include_non: bool) -> RowScore:
    ds = row.get("dataset", "")
    sc = RowScore(dropped_n=preds.n_dropped)
    if ds == "DCE":
        gold = {canonical_dce_rel(r) for r in gold_relations_dce(row)}
        pred = {canonical_dce_rel(r) for r in preds.kept}
    elif ds == "BIORED":
        index = build_biored_index(row)
        gold = gold_biored_pairs(row)
        pred = set()
        for r in preds.kept:
            i1 = match_biored_entity(r.get("entity1", ""), r.get("entity1_type", ""), index, fuzzy_min)
            i2 = match_biored_entity(r.get("entity2", ""), r.get("entity2_type", ""), index, fuzzy_min)
            if i1 is None or i2 is None or i1 == i2:
                sc.mapping_failures += 1
                continue
            pred.add((tuple(sorted((i1, i2))), r.get("relation", "")))
    else:
        sort_pair = (ds == "DDI") and (not ddi_directional)
        gold = {canonical_pair_rel(r, sort_pair) for r in gold_relations_pairwise(row, include_non)}
        pred = {canonical_pair_rel(r, sort_pair) for r in preds.kept}
    sc.gold_n, sc.pred_n = len(gold), len(pred)
    sc.tp, sc.fp, sc.fn = len(pred & gold), len(pred - gold), len(gold - pred)
    if strict:
        sc.fp += preds.n_dropped + sc.mapping_failures
    return sc


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def aggregate(scores: Sequence[RowScore]) -> Dict[str, Any]:
    tp = sum(s.tp for s in scores); fp = sum(s.fp for s in scores); fn = sum(s.fn for s in scores)
    out = prf(tp, fp, fn)
    out.update({"tp": tp, "fp": fp, "fn": fn, "n_rows": len(scores)})
    return out


def bootstrap_ci(scores: Sequence[RowScore], n: int, seed: int, alpha: float = 0.05):
    """Percentile CI on micro P/R/F1, resampling ROWS (relations within a row
    are not independent, so rows are the correct unit)."""
    if not scores or n <= 0:
        return {}
    rng = random.Random(seed)
    ps, rs, fs, k = [], [], [], len(scores)
    for _ in range(n):
        m = aggregate([scores[rng.randrange(k)] for _ in range(k)])
        ps.append(m["precision"]); rs.append(m["recall"]); fs.append(m["f1"])
    lo_i, hi_i = int(alpha / 2 * n), int((1 - alpha / 2) * n) - 1
    out = {}
    for name, vals in (("precision", ps), ("recall", rs), ("f1", fs)):
        vals.sort()
        out[name] = [round(vals[max(0, lo_i)], 4), round(vals[min(n - 1, hi_i)], 4)]
    return out


# =========================== prompting ================================

def format_text_block(text: str, context_text: str, context_mode: str) -> str:
    if context_mode == "with_context" and (context_text or "").strip():
        return f"TEXT:\n{text or ''}\n\nCONTEXT:\n{context_text}"
    return f"TEXT:\n{text or ''}"


def format_oracle_hint(row: Dict[str, Any]) -> str:
    """Gold entity mentions as hints, with an explicit instruction for EVERY
    dataset. Previously only BIORED.txt told the model what this block was."""
    ds = row.get("dataset", "")
    ents = row.get("entities", []) or []
    if not ents:
        return ""
    lines: List[str] = []
    if ds == "DCE":
        names = [normalize_space(e.get("text") or (e.get("surface_forms") or [""])[0])
                 for e in ents]
        names = [n for n in names if n]
        if not names:
            return ""
        lines.append("ALLOWED DRUG MENTIONS:")
        lines += [f"- {n}" for n in dict.fromkeys(names)]
    elif ds == "BIORED":
        lines.append("ALLOWED BIORED ENTITIES:")
        for e in ents:
            etype = normalize_type_for_dataset("BIORED", e.get("type", ""))
            if etype not in BIORED_ALLOWED_TYPES:
                continue
            forms = list(e.get("surface_forms", []) or [])
            if e.get("text"):
                forms.append(e["text"])
            forms = [normalize_space(f) for f in forms if f]
            if forms:
                lines.append(f"- {etype}: " + " | ".join(dict.fromkeys(forms)))
    else:
        by_type: Dict[str, List[str]] = defaultdict(list)
        for e in ents:
            txt = normalize_space(e.get("text") or (e.get("surface_forms") or [""])[0])
            if txt:
                by_type[normalize_type_for_dataset(ds, e.get("type", ""))].append(txt)
        if not by_type:
            return ""
        lines.append("ALLOWED ENTITY MENTIONS:")
        for t, names in by_type.items():
            lines.append(f"- {t}: " + " | ".join(dict.fromkeys(names)))
    if len(lines) <= 1:
        return ""
    lines += ["", "Use ONLY entities from this list, copying one listed surface form "
                  "exactly. Do not introduce entities that are not listed."]
    return "\n".join(lines)


def build_user_prompt(row, templates, context_mode: str, input_mode: str) -> str:
    block = format_text_block(row.get("text", ""), row.get("context_text", ""), context_mode)
    if input_mode == "oracle_entities":
        hint = format_oracle_hint(row)
        if hint:
            block = f"{block}\n\n{hint}"
    return templates[row.get("dataset", "")].format(text_block=block)


# ============================ backends ================================

def render_chat_template(tokenizer, system_prompt: str, user_prompt: str,
                         no_system_prompt: bool = False) -> str:
    if no_system_prompt:
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)


class BaseBackend:
    def __init__(self, system_prompt: str, args):
        self.system_prompt, self.args = system_prompt, args

    def generate(self, prompts: Sequence[str]) -> List[str]:
        raise NotImplementedError

    def close(self):
        pass


class VLLMBackend(BaseBackend):
    def __init__(self, system_prompt, args):
        super().__init__(system_prompt, args)
        from vllm import LLM
        from transformers import AutoTokenizer
        LOGGER.info("Loading tokenizer %s", args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        LOGGER.info("Loading vLLM engine (tp=%d, max_model_len=%d, enforce_eager=%s)",
                    args.tensor_parallel_size, args.max_model_len, args.enforce_eager)
        self.llm = LLM(model=args.model_path, dtype=args.dtype,
                       tensor_parallel_size=args.tensor_parallel_size,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       max_model_len=args.max_model_len, trust_remote_code=True,
                       seed=args.seed, enforce_eager=args.enforce_eager)

    def generate(self, prompts):
        from vllm import SamplingParams
        rendered = [render_chat_template(self.tokenizer, self.system_prompt, p,
                                         self.args.no_system_prompt) for p in prompts]
        kw = {"max_tokens": self.args.max_new_tokens, "n": 1,
              "temperature": self.args.temperature if self.args.do_sample else 0.0}
        if self.args.do_sample:
            kw["top_p"] = self.args.top_p
            kw["seed"] = self.args.seed
        try:
            params = SamplingParams(**kw)
        except TypeError:
            kw.pop("seed", None)
            params = SamplingParams(**kw)
        outs = self.llm.generate(rendered, params)
        res = [""] * len(rendered)
        for i, o in enumerate(outs):
            try:
                pos = int(o.request_id)
            except (AttributeError, TypeError, ValueError):
                pos = i
            if not 0 <= pos < len(res):
                pos = i
            res[pos] = o.outputs[0].text.strip() if o.outputs else ""
        return res


class HFBackend(BaseBackend):
    def __init__(self, system_prompt, args):
        super().__init__(system_prompt, args)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        dmap = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32, "auto": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dmap.get(args.dtype, torch.float16),
            device_map="auto", trust_remote_code=True)
        self.model.eval()

    def generate(self, prompts):
        torch = self.torch
        torch.manual_seed(self.args.seed)
        out = []
        for p in prompts:
            text = render_chat_template(self.tokenizer, self.system_prompt, p,
                                        self.args.no_system_prompt)
            inp = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            kw = {"max_new_tokens": self.args.max_new_tokens,
                  "do_sample": self.args.do_sample,
                  "pad_token_id": self.tokenizer.eos_token_id}
            if self.args.do_sample:
                kw["temperature"] = self.args.temperature
                kw["top_p"] = self.args.top_p
            with torch.no_grad():
                g = self.model.generate(**inp, **kw)
            out.append(self.tokenizer.decode(g[0][inp["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip())
        return out


def build_openai_messages(system_prompt: str, user_prompt: str,
                          no_system_prompt: bool = False) -> List[Dict[str, str]]:
    """Pure, network-free message construction -- unit-testable in isolation."""
    if no_system_prompt:
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


def call_with_retries(fn, max_retries: int, base_delay: float, sleep_fn=None,
                      retry_exceptions=(Exception,)) -> Tuple[Optional[Any], Optional[str]]:
    """Generic exponential-backoff retry wrapper, unit-testable with a fake
    flaky `fn` and a fake `sleep_fn` (no real network or sleeping needed).
    Returns (result, None) on success or (None, error_string) after giving up.
    """
    sleep = sleep_fn or (lambda _s: None)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return fn(), None
        except retry_exceptions as exc:
            last_err = str(exc)
            if attempt < max_retries:
                sleep(base_delay * (2 ** attempt))
    return None, last_err


class APIBackend(BaseBackend):
    """OpenAI-compatible chat-completions backend.

    Targets OpenAI's own API (gpt-4.1, gpt-4o, o-series, ...) and any provider
    exposing an OpenAI-compatible /chat/completions endpoint via
    --api_base_url (Azure OpenAI's compatible route, OpenRouter, Together,
    Fireworks, a local `vllm serve`, etc.). Anthropic's native Messages API
    uses a different request/response schema and is not covered here.

    Requests run concurrently via a thread pool (--api_concurrency), since the
    OpenAI SDK is a per-request call, not a batched engine like vLLM's.
    Reasoning-trace stripping still applies uniformly to whatever text comes
    back, so an OpenAI-compatible endpoint serving an open-reasoning model is
    still handled correctly downstream.
    """

    def __init__(self, system_prompt: str, args):
        super().__init__(system_prompt, args)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'api' backend requires the openai package: pip install openai") from exc
        import os
        key = os.environ.get(args.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {args.api_key_env} is not set. "
                f"export {args.api_key_env}=... before running with --backend api.")
        self.client = OpenAI(api_key=key, base_url=args.api_base_url or None)
        self.model = args.api_model or args.model_path
        if not self.model:
            raise RuntimeError("--api_model (or --model_path) must name a model, e.g. gpt-4.1.")
        LOGGER.info("Using API backend: model=%s base_url=%s", self.model,
                    args.api_base_url or "https://api.openai.com/v1 (default)")

    def _one(self, prompt: str) -> str:
        messages = build_openai_messages(self.system_prompt, prompt, self.args.no_system_prompt)

        def call():
            kw = {"model": self.model, "messages": messages,
                  "timeout": self.args.api_request_timeout,
                  "temperature": self.args.temperature if self.args.do_sample else 0.0}
            resp = self.client.chat.completions.create(**kw)
            return resp.choices[0].message.content or ""

        import time
        result, err = call_with_retries(call, self.args.api_max_retries,
                                        base_delay=1.0, sleep_fn=time.sleep)
        if err is not None:
            LOGGER.warning("API call failed after retries: %s", err)
            return ""
        return result

    def generate(self, prompts: Sequence[str]) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.args.api_concurrency) as pool:
            return list(pool.map(self._one, prompts))


def estimate_tokens_rough(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token for English),
    used only to keep a single Batch API submission under the org's
    enqueued-token cap. This is deliberately conservative-ish, not exact --
    if the org's real limit is still hit, lower --batch_max_enqueued_tokens.
    """
    return max(1, (len(text) + 3) // 4)


def estimate_request_tokens(system_prompt: str, user_prompt: str, max_new_tokens: int) -> int:
    """Estimate the tokens one batch request line contributes to the
    enqueued-token count: prompt tokens (system+user, plus a small per-message
    overhead) plus the requested completion budget, since OpenAI counts
    against the cap as soon as a request is enqueued, before it's known how
    many completion tokens will actually be used.
    """
    prompt_tokens = estimate_tokens_rough(system_prompt) + estimate_tokens_rough(user_prompt) + 16
    return prompt_tokens + max_new_tokens


def build_batch_request_line(custom_id: str, model: str, system_prompt: str, user_prompt: str,
                             no_system_prompt: bool, temperature: float) -> Dict[str, Any]:
    """Pure, network-free construction of one Batch API JSONL line."""
    messages = build_openai_messages(system_prompt, user_prompt, no_system_prompt)
    body = {"model": model, "messages": messages, "temperature": temperature}
    return {"custom_id": custom_id, "method": "POST",
            "url": "/v1/chat/completions", "body": body}


def parse_batch_output(output_content: str, n: int) -> List[str]:
    """Pure, network-free parsing of a downloaded Batch API output file into
    a results list aligned with the original prompt order. Lines with a
    non-200 status, malformed custom_id, or no choices leave that slot "".
    """
    import json as _json
    results = [""] * n
    for line in output_content.splitlines():
        if not line.strip():
            continue
        try:
            rec = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        custom_id = rec.get("custom_id", "")
        try:
            idx = int(custom_id.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if not 0 <= idx < n:
            continue
        resp = rec.get("response") or {}
        if resp.get("status_code") == 200:
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if choices:
                results[idx] = choices[0].get("message", {}).get("content") or ""
    return results


def chunk_indices_by_token_budget(n: int, per_request_tokens: Sequence[int],
                                  token_budget: int, max_requests: int) -> List[Tuple[int, int]]:
    """Split range(n) into contiguous [start, end) chunks such that each
    chunk's total estimated tokens stays under token_budget and each chunk
    has at most max_requests requests. Pure and unit-testable.

    A single request whose own estimate exceeds token_budget still gets its
    own one-request chunk rather than looping forever -- the real API call
    may still fail for that one row, but it won't block the rest of the run.
    """
    if n <= 0:
        return []
    chunks: List[Tuple[int, int]] = []
    start = 0
    running = 0
    count = 0
    for i in range(n):
        t = per_request_tokens[i]
        would_exceed_tokens = running + t > token_budget and count > 0
        would_exceed_count = count >= max_requests
        if would_exceed_tokens or would_exceed_count:
            chunks.append((start, i))
            start = i
            running = 0
            count = 0
        running += t
        count += 1
    chunks.append((start, n))
    return chunks


class BatchAPIBackend(BaseBackend):
    """OpenAI Batch API backend: async, typically ~50% cheaper than the
    synchronous endpoint, up to a 24h completion window.

    Unlike APIBackend, this makes as few jobs as possible: prompts are
    submitted as ONE batch job unless the estimated enqueued tokens would
    exceed --batch_max_enqueued_tokens (OpenAI enforces this per-model, org-
    wide, across ALL in-flight batches -- see
    https://platform.openai.com/docs/guides/batch). When a run is too big
    for one job, prompts are split into multiple contiguous chunks, each
    submitted and polled to completion IN SEQUENCE (submitting the next
    chunk before the previous one clears would just hit the same limit
    again, since it's cumulative across in-flight jobs, not per-job).

    RESUMABILITY. A 24h window is too long to hold an interactive
    connection or a fragile SSH session open. --batch_state_file records
    every chunk's batch_id and status (submitted/completed) as the run
    progresses. Rerunning with the same --batch_state_file resumes: chunks
    already marked completed are re-downloaded instead of resubmitted,
    a chunk left mid-flight is re-polled, and any remaining chunks are
    submitted fresh. --batch_resume_id is kept as a shorthand for the
    common single-chunk case (small run, or already under the token cap).

    LIMITATION. With --input_mode all this backend is used twice (once per
    condition), sequentially -- text_only's job(s) must finish before
    oracle_entities' job(s) are submitted. If the process dies while waiting
    on the SECOND condition, the first condition's already-computed results
    are not yet written to disk (nothing is written until both conditions
    finish) and are lost. For a long batch run, prefer two separate
    invocations with --input_mode text_only and --input_mode
    oracle_entities, each with its own --batch_state_file, so a crash only
    costs one condition.
    """

    def __init__(self, system_prompt: str, args):
        super().__init__(system_prompt, args)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'api' backend requires the openai package: pip install openai") from exc
        import os
        key = os.environ.get(args.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {args.api_key_env} is not set. "
                f"export {args.api_key_env}=... before running with --backend api.")
        self.client = OpenAI(api_key=key, base_url=args.api_base_url or None)
        self.model = args.api_model or args.model_path
        if not self.model:
            raise RuntimeError("--api_model (or --model_path) must name a model, e.g. gpt-4.1-mini.")
        LOGGER.info("Using Batch API backend: model=%s completion_window=%s "
                    "max_enqueued_tokens=%d", self.model, args.batch_completion_window,
                    args.batch_max_enqueued_tokens)

    # -- state file helpers -------------------------------------------------

    def _load_state(self, n_prompts: int) -> Optional[Dict[str, Any]]:
        if not self.args.batch_state_file:
            return None
        p = Path(self.args.batch_state_file)
        if not p.exists():
            return None
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if state.get("model") != self.model or state.get("n_prompts") != n_prompts:
            LOGGER.warning("Ignoring --batch_state_file %s: model/n_prompts don't match "
                          "this run (found model=%s n_prompts=%s).",
                          self.args.batch_state_file, state.get("model"), state.get("n_prompts"))
            return None
        return state

    def _save_state(self, state: Dict[str, Any]) -> None:
        if not self.args.batch_state_file:
            return
        with open(self.args.batch_state_file, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

    # -- single chunk submit / poll / download ------------------------------

    def _submit_chunk(self, prompts: Sequence[str], start: int) -> Tuple[str, str]:
        import json as _json
        import tempfile
        temperature = self.args.temperature if self.args.do_sample else 0.0
        lines = [build_batch_request_line(f"req-{start + i}", self.model, self.system_prompt, p,
                                          self.args.no_system_prompt, temperature)
                 for i, p in enumerate(prompts)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as fh:
            for line in lines:
                fh.write(_json.dumps(line, ensure_ascii=False) + "\n")
            batch_input_path = fh.name

        LOGGER.info("Uploading %d requests for batch submission (rows %d-%d)...",
                    len(lines), start, start + len(lines) - 1)
        with open(batch_input_path, "rb") as fh:
            uploaded = self.client.files.create(file=fh, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=uploaded.id, endpoint="/v1/chat/completions",
            completion_window=self.args.batch_completion_window,
            metadata={"source": "run_e2e_llm.py"})
        LOGGER.info("Batch job created: id=%s status=%s", batch.id, batch.status)
        return batch.id, uploaded.id

    def _poll(self, batch_id: str) -> Any:
        import time
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = self.client.batches.retrieve(batch_id)
            counts = batch.request_counts
            done = getattr(counts, "completed", 0) if counts else 0
            total = getattr(counts, "total", 0) if counts else 0
            failed = getattr(counts, "failed", 0) if counts else 0
            LOGGER.info("Batch %s: status=%s completed=%s/%s failed=%s",
                        batch_id, batch.status, done, total, failed)
            if batch.status in terminal:
                return batch
            time.sleep(self.args.batch_poll_interval)

    def _download_chunk(self, batch, n: int) -> List[str]:
        if batch.status != "completed":
            LOGGER.error("Batch %s ended with status=%s (not 'completed'). "
                        "Returning what succeeded; everything else scores as a "
                        "parse failure downstream.", batch.id, batch.status)
        if batch.error_file_id:
            err_content = self.client.files.content(batch.error_file_id).text
            n_err = sum(1 for line in err_content.splitlines() if line.strip())
            LOGGER.warning("Batch %s: %d requests failed (error file %s).",
                          batch.id, n_err, batch.error_file_id)
        results = [""] * n
        if batch.output_file_id:
            out_content = self.client.files.content(batch.output_file_id).text
            results = parse_batch_output(out_content, n)
        return results

    # -- public entry point --------------------------------------------------

    def generate(self, prompts: Sequence[str]) -> List[str]:
        n = len(prompts)
        results = [""] * n

        # Legacy shorthand: caller already knows the single job's batch_id
        # (small run, or a run known to fit under the token cap).
        if self.args.batch_resume_id:
            LOGGER.info("Reattaching to existing batch job: %s", self.args.batch_resume_id)
            batch = self._poll(self.args.batch_resume_id)
            return self._download_chunk(batch, n)

        per_req_tokens = [estimate_request_tokens(self.system_prompt, p, self.args.max_new_tokens)
                          for p in prompts]
        chunk_ranges = chunk_indices_by_token_budget(
            n, per_req_tokens, self.args.batch_max_enqueued_tokens,
            self.args.batch_max_requests_per_job)

        if len(chunk_ranges) == 1:
            LOGGER.info("Submitting all %d prompts as a single batch job "
                       "(estimated tokens fit under --batch_max_enqueued_tokens=%d).",
                       n, self.args.batch_max_enqueued_tokens)
        else:
            LOGGER.info("Estimated enqueued tokens for %d prompts exceed "
                       "--batch_max_enqueued_tokens=%d; splitting into %d sequential "
                       "batch jobs (each still runs within its own %s window; only "
                       "submission is sequential, to stay under the org-wide "
                       "in-flight token cap).", n, self.args.batch_max_enqueued_tokens,
                       len(chunk_ranges), self.args.batch_completion_window)

        state = self._load_state(n) or {"model": self.model, "n_prompts": n, "chunks": {}}
        chunk_state = state.setdefault("chunks", {})

        for ci, (start, end) in enumerate(chunk_ranges):
            key = str(start)
            rec = chunk_state.get(key)
            chunk_prompts = prompts[start:end]

            if rec and rec.get("status") == "completed" and rec.get("results_cached"):
                LOGGER.info("Chunk %d/%d (rows %d-%d): resuming, already completed "
                          "(batch_id=%s).", ci + 1, len(chunk_ranges), start, end - 1,
                          rec.get("batch_id"))
                results[start:end] = rec["results_cached"]
                continue

            if rec and rec.get("batch_id"):
                batch_id = rec["batch_id"]
                LOGGER.info("Chunk %d/%d (rows %d-%d): resuming submitted job %s.",
                          ci + 1, len(chunk_ranges), start, end - 1, batch_id)
            else:
                batch_id, input_file_id = self._submit_chunk(chunk_prompts, start)
                chunk_state[key] = {"batch_id": batch_id, "input_file_id": input_file_id,
                                    "start": start, "end": end, "status": "submitted"}
                self._save_state(state)

            batch = self._poll(batch_id)
            chunk_results = self._download_chunk(batch, len(chunk_prompts))
            results[start:end] = chunk_results

            chunk_state[key]["status"] = "completed" if batch.status == "completed" else batch.status
            chunk_state[key]["results_cached"] = chunk_results
            self._save_state(state)

            n_empty = sum(1 for r in chunk_results if not r)
            LOGGER.info("Chunk %d/%d (rows %d-%d): %d/%d results recovered (%d empty/failed).",
                        ci + 1, len(chunk_ranges), start, end - 1,
                        len(chunk_results) - n_empty, len(chunk_results), n_empty)

        n_empty_total = sum(1 for r in results if not r)
        LOGGER.info("Batch run: %d/%d results recovered across %d job(s) (%d empty/failed).",
                    n - n_empty_total, n, len(chunk_ranges), n_empty_total)
        return results


class EchoBackend(BaseBackend):
    def __init__(self, system_prompt, args, responses=None):
        super().__init__(system_prompt, args)
        self.responses = responses or ['{"relations": []}']

    def generate(self, prompts):
        return [self.responses[i % len(self.responses)] for i in range(len(prompts))]


# ========================= run one condition ==========================

def run_condition(rows, backend, templates, args, input_mode: str):
    LOGGER.info("=== input_mode=%s context_mode=%s ===", input_mode, args.context_mode)
    prompts = [build_user_prompt(r, templates, args.context_mode, input_mode) for r in rows]
    raws: List[str] = []
    if isinstance(backend, BatchAPIBackend):
        # BatchAPIBackend decides internally whether this fits in one job or
        # must be split into sequential chunks to stay under the org's
        # enqueued-token cap; --batch_size is not used here either way.
        raws = backend.generate(prompts)
    else:
        for s in range(0, len(prompts), args.batch_size):
            raws.extend(backend.generate(prompts[s:s + args.batch_size]))
            LOGGER.info("  %d/%d generated", min(s + args.batch_size, len(prompts)), len(prompts))

    parse_errors: Counter = Counter()
    drop_reasons: Counter = Counter()
    per_ds = defaultdict(lambda: {"lenient": [], "strict": []})
    ddi_unordered: List[RowScore] = []
    predictions: List[Dict[str, Any]] = []
    strata = defaultdict(lambda: defaultdict(list))

    for row, raw in zip(rows, raws):
        ds = row.get("dataset", "")
        pred_obj, perr = parse_prediction(raw)
        if perr:
            parse_errors[perr] += 1
        norm = normalize_predictions(ds, pred_obj["relations"])
        for _, reason in norm.dropped:
            drop_reasons[f"{ds}:{reason}"] += 1

        common = dict(fuzzy_min=args.fuzzy_min_score, include_non=args.ddi_include_non)
        s_len = score_row(row, norm, strict=False, ddi_directional=args.ddi_directional, **common)
        s_str = score_row(row, norm, strict=True, ddi_directional=args.ddi_directional, **common)
        per_ds[ds]["lenient"].append(s_len)
        per_ds[ds]["strict"].append(s_str)
        if ds == "DDI":
            ddi_unordered.append(score_row(row, norm, strict=False,
                                           ddi_directional=False, **common))

        gn = s_len.gold_n
        b = "0" if gn == 0 else "1" if gn == 1 else "2-4" if gn <= 4 else "5-9" if gn <= 9 else "10+"
        strata[ds][f"gold_relations={b}"].append(s_len)
        tl = len(row.get("text", "") or "")
        tb = "<500" if tl < 500 else "500-1499" if tl < 1500 else "1500+"
        strata[ds][f"text_chars={tb}"].append(s_len)

        if args.save_predictions:
            predictions.append({
                "benchmark_id": row.get("benchmark_id"), "dataset": ds,
                "input_mode": input_mode, "context_mode": args.context_mode,
                "raw_output": raw if args.save_raw else None,
                "parsed_relations": norm.kept,
                "dropped_predictions": [{"relation": d, "reason": r} for d, r in norm.dropped],
                "parse_error": perr, "tp": s_len.tp, "fp": s_len.fp, "fn": s_len.fn,
                "gold_n": s_len.gold_n, "pred_n": s_len.pred_n,
                "mapping_failures": s_len.mapping_failures})

    def block(mode: str) -> Dict[str, Any]:
        by_ds = {ds: aggregate(v[mode]) for ds, v in per_ds.items()}
        all_scores = [s for v in per_ds.values() for s in v[mode]]
        present = [d for d in ALL_DATASETS if d in by_ds]
        macro = ({k: round(statistics.mean([by_ds[d][k] for d in present]), 4)
                  for k in ("precision", "recall", "f1")} if present else {})
        out = {"per_dataset": by_ds, "micro_overall": aggregate(all_scores),
               "macro_over_datasets": macro,
               "_note": ("micro_overall pools tp/fp/fn across all rows (instance-weighted). "
                         "macro_over_datasets is the unweighted mean of the four per-dataset "
                         "scores -- this is what Tables III/IV report.")}
        if args.bootstrap > 0:
            out["micro_ci95"] = bootstrap_ci(all_scores, args.bootstrap, args.seed)
            out["per_dataset_ci95"] = {ds: bootstrap_ci(v[mode], args.bootstrap, args.seed)
                                       for ds, v in per_ds.items()}
        return out

    summary = {
        "input_mode": input_mode, "context_mode": args.context_mode, "n_rows": len(rows),
        "decoding": {"do_sample": args.do_sample, "deterministic": not args.do_sample,
                     "temperature": args.temperature if args.do_sample else 0.0,
                     "max_new_tokens": args.max_new_tokens, "seed": args.seed},
        "scoring_config": {"ddi_directional": args.ddi_directional,
                           "ddi_include_non": args.ddi_include_non,
                           "biored_fuzzy_min_score": args.fuzzy_min_score},
        "failures": {"parse_errors": dict(parse_errors),
                     "parse_error_total": sum(parse_errors.values()),
                     "dropped_predictions_by_reason": dict(drop_reasons.most_common()),
                     "dropped_predictions_total": sum(drop_reasons.values()),
                     "biored_mapping_failures": sum(s.mapping_failures
                                                    for v in per_ds.values()
                                                    for s in v["lenient"])},
        "lenient": block("lenient"), "strict": block("strict"),
        "_scoring_note": ("lenient drops schema-invalid predictions before scoring (the "
                          "previous pipeline's behaviour); strict counts them as false "
                          "positives. Report both."),
    }
    if ddi_unordered:
        summary["ddi_argument_order_sensitivity"] = {
            "ordered_e1_e2": aggregate(per_ds["DDI"]["lenient"]),
            "unordered_sorted_pair": aggregate(ddi_unordered),
            "_note": ("The manuscript states DDI matching preserves ordered argument roles; "
                      "the previous code sorted the pair. Both are reported.")}
    summary["diagnostics"] = {ds: {k: aggregate(v) for k, v in sorted(b.items())}
                              for ds, b in strata.items()}
    return summary, predictions


# ============================= self-test ==============================

def self_test() -> int:
    fails = 0

    def check(name, cond, extra=""):
        nonlocal fails
        ok = bool(cond)
        fails += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'' if ok else '  ' + extra}")

    j = ('{"relations": [{"entity1":"a","entity1_type":"CHEMICAL","entity2":"b",'
         '"entity2_type":"GENE","relation":"CPR:4"}]}')
    for nm, raw, expect in [("plain JSON", j, True),
                            ("closed <think>", '<think>maybe {"x":1}</think>' + j, True),
                            ("fenced after think", '<think>r</think>\n```json\n' + j + '\n```', True),
                            ("unclosed <think>", '<think>reasoning {a', False)]:
        obj, err = extract_json_block(raw)
        check(f"parse: {nm}", (obj is not None) == expect, f"err={err}")
    _, err = extract_json_block('{"relations": [{"entity1": "a"')
    check("truncated JSON flagged", err == "unbalanced_json_likely_truncated", f"got {err}")

    r, _ = chemprot_reorder({"entity1": "COX-2", "entity1_type": "GENE",
                             "entity2": "aspirin", "entity2_type": "CHEMICAL", "relation": "CPR:4"})
    check("chemprot reorders to chemical-first", r and r["entity1"] == "aspirin")
    r2, why = chemprot_reorder({"entity1": "aspirin", "entity1_type": "CHEMICAL",
                                "entity2": "ibuprofen", "entity2_type": "CHEMICAL",
                                "relation": "CPR:4"})
    check("chemprot rejects CHEMICAL-CHEMICAL", r2 is None and why == "invalid_type_pairing")

    row = {"dataset": "CHEMPROT", "text": "t",
           "entities": [{"entity_id": "T1", "text": "aspirin", "type": "CHEMICAL"},
                        {"entity_id": "T2", "text": "COX-2", "type": "GENE-Y"}],
           "relations": [{"label": "CPR:4", "arguments": [{"role": "arg1", "entity_id": "T1"},
                                                          {"role": "arg2", "entity_id": "T2"}]}]}
    preds = normalize_predictions("CHEMPROT", [
        {"entity1": "aspirin", "entity1_type": "CHEMICAL", "entity2": "COX-2",
         "entity2_type": "GENE", "relation": "CPR:4"},
        {"entity1": "aspirin", "entity1_type": "CHEMICAL", "entity2": "ibuprofen",
         "entity2_type": "CHEMICAL", "relation": "CPR:4"}])
    check("invalid prediction separated", len(preds.kept) == 1 and preds.n_dropped == 1)
    kw = dict(fuzzy_min=DEFAULT_FUZZY_MIN_SCORE, include_non=False, ddi_directional=True)
    sl = score_row(row, preds, strict=False, **kw)
    ss = score_row(row, preds, strict=True, **kw)
    check("lenient precision 1.0", prf(sl.tp, sl.fp, sl.fn)["precision"] == 1.0)
    check("strict precision 0.5", prf(ss.tp, ss.fp, ss.fn)["precision"] == 0.5,
          f"got {prf(ss.tp, ss.fp, ss.fn)}")

    ddi = {"dataset": "DDI", "text": "t",
           "entities": [{"entity_id": "e0", "text": "drugA", "type": "DRUG"},
                        {"entity_id": "e1", "text": "drugB", "type": "DRUG"}],
           "relations": [{"label": "EFFECT", "arguments": [{"role": "e1", "entity_id": "e0"},
                                                           {"role": "e2", "entity_id": "e1"}]}]}
    swapped = normalize_predictions("DDI", [{"entity1": "drugB", "entity1_type": "DRUG",
                                             "entity2": "drugA", "entity2_type": "DRUG",
                                             "relation": "EFFECT"}])
    c = dict(fuzzy_min=DEFAULT_FUZZY_MIN_SCORE, include_non=False)
    check("DDI ordered: swapped args miss",
          score_row(ddi, swapped, strict=False, ddi_directional=True, **c).tp == 0)
    check("DDI unordered: swapped args match",
          score_row(ddi, swapped, strict=False, ddi_directional=False, **c).tp == 1)

    ddi_non = dict(ddi)
    ddi_non["relations"] = [{"label": "NON", "arguments": [{"role": "e1", "entity_id": "e0"},
                                                           {"role": "e2", "entity_id": "e1"}]}]
    check("DDI NON excluded from gold", len(gold_relations_pairwise(ddi_non, False)) == 0)
    check("DDI NON included on request", len(gold_relations_pairwise(ddi_non, True)) == 1)
    check("predicted NON dropped", normalize_predictions("DDI", [
        {"entity1": "drugA", "entity1_type": "DRUG", "entity2": "drugB",
         "entity2_type": "DRUG", "relation": "NON"}]).n_dropped == 1)

    micro = aggregate([RowScore(tp=90, fp=10, fn=10), RowScore(tp=1, fp=9, fn=9)])
    macro = statistics.mean([aggregate([RowScore(tp=90, fp=10, fn=10)])["f1"],
                             aggregate([RowScore(tp=1, fp=9, fn=9)])["f1"]])
    check("micro != macro when datasets are imbalanced", abs(micro["f1"] - macro) > 0.1,
          f"micro={micro['f1']} macro={macro:.4f}")

    ci = bootstrap_ci([RowScore(tp=1, fp=1, fn=1) for _ in range(50)], 200, 0)
    check("bootstrap returns CI", "f1" in ci and len(ci["f1"]) == 2)
    for ds, r_ in [("CHEMPROT", row), ("DDI", ddi)]:
        check(f"oracle hint emitted for {ds}", "ALLOWED" in format_oracle_hint(r_))

    # ---- API backend (network-free unit tests) ----
    msgs = build_openai_messages("SYS", "USER")
    check("api messages: system+user by default",
          msgs == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}])
    msgs2 = build_openai_messages("SYS", "USER", no_system_prompt=True)
    check("api messages: merged when no_system_prompt",
          msgs2 == [{"role": "user", "content": "SYS\n\nUSER"}])

    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"
    sleeps = []
    result, err = call_with_retries(flaky, max_retries=5, base_delay=1.0,
                                    sleep_fn=lambda s: sleeps.append(s))
    check("retry wrapper succeeds after transient failures", result == "ok" and err is None)
    check("retry wrapper backs off exponentially", sleeps == [1.0, 2.0], str(sleeps))

    def always_fails():
        raise RuntimeError("permanent")
    result2, err2 = call_with_retries(always_fails, max_retries=2, base_delay=0.0,
                                      sleep_fn=lambda s: None)
    check("retry wrapper gives up and reports the error",
          result2 is None and err2 == "permanent")

    # ---- Batch API (network-free unit tests) ----
    line = build_batch_request_line("req-3", "gpt-4.1-mini", "SYS", "USER", False, 0.0)
    check("batch request line: custom_id preserved", line["custom_id"] == "req-3")
    check("batch request line: correct endpoint", line["url"] == "/v1/chat/completions")
    check("batch request line: model/messages/temperature present",
          line["body"]["model"] == "gpt-4.1-mini"
          and line["body"]["messages"] == build_openai_messages("SYS", "USER")
          and line["body"]["temperature"] == 0.0)

    fake_output = "\n".join([
        json.dumps({"custom_id": "req-0", "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": '{"relations": []}'}}]}}}),
        json.dumps({"custom_id": "req-1", "response": {"status_code": 429, "body": {}}}),
        json.dumps({"custom_id": "req-2", "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": '{"relations": [{"relation":"CPR:4"}]}'}}]}}}),
        "",  # blank line, must be skipped
        json.dumps({"custom_id": "not-an-index", "response": {"status_code": 200, "body": {}}}),
    ])
    parsed = parse_batch_output(fake_output, n=3)
    check("batch output: successful line recovered", parsed[0] == '{"relations": []}')
    check("batch output: failed status_code -> empty slot", parsed[1] == "")
    check("batch output: order preserved by custom_id, not file order",
          parsed[2] == '{"relations": [{"relation":"CPR:4"}]}')
    check("batch output: malformed custom_id ignored without crashing", len(parsed) == 3)

    # ---- token-budget chunking (network-free unit tests) ----
    small = chunk_indices_by_token_budget(5, [100] * 5, token_budget=1000, max_requests=50000)
    check("chunking: everything fits in one chunk when under budget", small == [(0, 5)])

    tight = chunk_indices_by_token_budget(5, [300, 300, 300, 300, 300],
                                          token_budget=1000, max_requests=50000)
    check("chunking: splits when cumulative tokens exceed budget",
          len(tight) > 1 and tight[0][0] == 0 and tight[-1][1] == 5, str(tight))
    covered = set()
    for s, e in tight:
        covered.update(range(s, e))
    check("chunking: every index covered exactly once", covered == set(range(5)), str(tight))

    huge_one = chunk_indices_by_token_budget(2, [5000, 10], token_budget=1000, max_requests=50000)
    check("chunking: an oversized single request still gets its own chunk",
          huge_one[0] == (0, 1), str(huge_one))

    by_count = chunk_indices_by_token_budget(5, [1] * 5, token_budget=10 ** 9, max_requests=2)
    check("chunking: also splits on max_requests", by_count == [(0, 2), (2, 4), (4, 5)],
          str(by_count))

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


# =============================== CLI ==================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end / oracle-entity biomedical RE evaluation",
                                allow_abbrev=False)
    p.add_argument("--input_file", type=str, default="")
    p.add_argument("--output_dir", type=str, default="outputs/re_eval")
    p.add_argument("--prompt_dir", type=str, default="prompts")
    p.add_argument("--model_path", type=str, default="")

    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf", "echo", "api"],
                   help="'api' targets any OpenAI-compatible chat-completions endpoint "
                        "(OpenAI's gpt-4.1, gpt-4o, etc., Azure OpenAI, OpenRouter, Together, "
                        "a local `vllm serve` -- anything speaking the same schema). "
                        "Anthropic's native API is not OpenAI-schema-compatible and is not "
                        "covered.")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--enforce_eager", action="store_true", default=True)
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--no_system_prompt", action="store_true",
                   help="Merge system+user into one turn (replaces run_e2e_llm_nsp.py).")

    p.add_argument("--api_model", type=str, default="",
                   help="Model name for --backend api, e.g. gpt-4.1. Falls back to "
                        "--model_path if not given.")
    p.add_argument("--api_base_url", type=str, default="",
                   help="OpenAI-compatible endpoint URL. Empty = OpenAI's default.")
    p.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY",
                   help="Environment variable holding the API key. Never pass the key "
                        "on the command line.")
    p.add_argument("--api_concurrency", type=int, default=8,
                   help="Concurrent in-flight requests (the API is per-request, not "
                        "batched like vLLM). Tune to the provider's rate limit.")
    p.add_argument("--api_max_retries", type=int, default=5)
    p.add_argument("--api_request_timeout", type=float, default=60.0)

    p.add_argument("--api_mode", type=str, default="sync", choices=["sync", "batch"],
                   help="'sync' calls the standard chat-completions endpoint concurrently "
                        "(--api_concurrency). 'batch' submits the OpenAI Batch API: "
                        "typically ~50%% cheaper, up to a 24h completion window, as few "
                        "jobs as possible for the whole run instead of one request per "
                        "row (see --batch_max_enqueued_tokens if the run is large). Only "
                        "meaningful with --backend api.")
    p.add_argument("--batch_completion_window", type=str, default="24h",
                   help="OpenAI Batch API currently only supports '24h'.")
    p.add_argument("--batch_poll_interval", type=float, default=60.0,
                   help="Seconds between batch status checks while waiting.")
    p.add_argument("--batch_max_enqueued_tokens", type=int, default=1_800_000,
                   help="Safety budget (in estimated tokens) per batch job. OpenAI caps "
                        "total ENQUEUED tokens per model across all in-flight batch jobs "
                        "org-wide (commonly 2,000,000 -- check your org's limit on the "
                        "usage/limits page); a single job that would exceed this is "
                        "rejected with status=failed and no per-request errors. Keep this "
                        "comfortably under your org's actual cap. When a run's estimated "
                        "tokens exceed this, it is split into multiple sequential batch "
                        "jobs automatically.")
    p.add_argument("--batch_max_requests_per_job", type=int, default=45_000,
                   help="Also cap requests per job (OpenAI's hard limit is 50,000 "
                        "requests / 200MB per batch file); kept a bit under that limit.")
    p.add_argument("--batch_state_file", type=str, default="",
                   help="Where to save per-chunk {batch_id, status, ...} as the run "
                        "progresses. Empty = don't save (not recommended for --api_mode "
                        "batch given the long completion window). Rerunning with the "
                        "same file resumes: completed chunks are re-downloaded instead "
                        "of resubmitted, in-flight chunks are re-polled, and remaining "
                        "chunks are submitted fresh.")
    p.add_argument("--batch_resume_id", type=str, default="",
                   help="Reattach to a single already-submitted batch job instead of "
                        "submitting new one(s) -- shorthand for the common case where the "
                        "whole run fit in one job. Skips generation entirely and goes "
                        "straight to polling + downloading. Not used for a run that needed "
                        "multiple chunks -- use --batch_state_file to resume those.")

    p.add_argument("--context_mode", type=str, default="no_context",
                   choices=["no_context", "with_context"],
                   help="Controls ONLY whether context_text is appended. Independent of "
                        "--input_mode. context_text is model-generated for "
                        "CHEMPROT/DDI/BIORED in the shipped corpus.")
    p.add_argument("--input_mode", type=str, default="all",
                   choices=["text_only", "oracle_entities", "all"],
                   help="Controls ONLY whether gold entity mentions are appended.")
    p.add_argument("--dataset", type=str, default="all")
    p.add_argument("--annotation_level", type=str, default="all",
                   choices=["all", "mention-level", "entity-level"])
    p.add_argument("--max_examples", type=int, default=-1)

    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--do_sample", action="store_true",
                   help="Off by default: decoding is greedy and deterministic.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--scoring", type=str, default="both", choices=["lenient", "strict", "both"])
    p.add_argument("--ddi_directional", action="store_true", default=True)
    p.add_argument("--ddi_unordered", dest="ddi_directional", action="store_false")
    p.add_argument("--ddi_include_non", action="store_true")
    p.add_argument("--fuzzy_min_score", type=float, default=DEFAULT_FUZZY_MIN_SCORE)
    p.add_argument("--bootstrap", type=int, default=1000)

    p.add_argument("--save_predictions", action="store_true", default=True)
    p.add_argument("--no_save_predictions", dest="save_predictions", action="store_false")
    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--self_test", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.self_test:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return self_test()
    if not args.input_file:
        print("--input_file is required (or use --self_test).", file=sys.stderr)
        return 2
    if args.backend in {"vllm", "hf"} and not args.model_path:
        print("--model_path is required for vllm/hf backends.", file=sys.stderr)
        return 2
    if args.backend == "api" and not (args.api_model or args.model_path):
        print("--api_model (e.g. gpt-4.1) is required for --backend api.", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out_dir / "eval.log", encoding="utf-8")],
                        force=True)
    if args.max_new_tokens < 512:
        LOGGER.warning("max_new_tokens=%d is low. BioRED rows often need more; truncated "
                       "JSON fails to parse and silently scores as zero recall.",
                       args.max_new_tokens)

    rows = []
    with open(args.input_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.dataset != "all":
        keep = {d.strip().upper() for d in args.dataset.split(",")}
        rows = [r for r in rows if (r.get("dataset") or "").upper() in keep]
    if args.annotation_level != "all":
        rows = [r for r in rows if r.get("annotation_level") == args.annotation_level]
    if args.max_examples and args.max_examples > 0:
        rows = rows[:args.max_examples]
    LOGGER.info("Loaded %d rows.", len(rows))
    if not rows:
        LOGGER.error("No rows to evaluate.")
        return 1

    pdir = Path(args.prompt_dir)
    templates = {"system": (pdir / "system.txt").read_text(encoding="utf-8")}
    for ds in ALL_DATASETS:
        f = pdir / f"{ds}.txt"
        if f.exists():
            templates[ds] = f.read_text(encoding="utf-8")

    backend = (VLLMBackend(templates["system"], args) if args.backend == "vllm"
               else HFBackend(templates["system"], args) if args.backend == "hf"
               else (BatchAPIBackend(templates["system"], args) if args.api_mode == "batch"
                     else APIBackend(templates["system"], args)) if args.backend == "api"
               else EchoBackend(templates["system"], args))

    modes = ["text_only", "oracle_entities"] if args.input_mode == "all" else [args.input_mode]
    all_summaries = {}
    try:
        for mode in modes:
            summary, preds = run_condition(rows, backend, templates, args, mode)
            tag = f"{args.dataset}_{args.annotation_level}_{args.context_mode}_{mode}"
            with open(out_dir / f"summary_{tag}.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=2)
            if args.save_predictions:
                with open(out_dir / f"predictions_{tag}.jsonl", "w", encoding="utf-8") as fh:
                    for pr in preds:
                        fh.write(json.dumps(pr, ensure_ascii=False) + "\n")
            all_summaries[mode] = summary
            LOGGER.info("--- %s ---", mode)
            for scheme in (["lenient", "strict"] if args.scoring == "both" else [args.scoring]):
                m, mac = summary[scheme]["micro_overall"], summary[scheme]["macro_over_datasets"]
                LOGGER.info("  %-8s micro P=%.3f R=%.3f F1=%.3f | macro F1=%.3f",
                            scheme, m["precision"], m["recall"], m["f1"], mac.get("f1", 0))
            LOGGER.info("  parse errors=%d dropped preds=%d",
                        summary["failures"]["parse_error_total"],
                        summary["failures"]["dropped_predictions_total"])
    finally:
        backend.close()

    if len(modes) > 1:
        tag = f"{args.dataset}_{args.annotation_level}_{args.context_mode}_all"
        with open(out_dir / f"summary_{tag}.json", "w", encoding="utf-8") as fh:
            json.dump(all_summaries, fh, ensure_ascii=False, indent=2)
    LOGGER.info("Outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

  #  python run_e2e_llm.py --input_file data/unified_RE_benchmark.jsonl \
  #--model_path /u/erdos/csga2/aalfatemi/LLM_MSGCNET/NSMA/Models/Mistral-Small-3.1-24B-Instruct-2503 --prompt_dir prompts_reletion_extraction --backend vllm \
  #--tensor_parallel_size 2 --input_mode all --scoring both --output_dir outputs/Mistral-Small-31-24B --batch_size 64

 # python scripts/run_e2e_llm.py --input_file data/unified_RE_benchmark.jsonl \
 #--backend api --api_model gpt-4.1-mini --api_mode sync --prompt_dir prompts_reletion_extraction \
 # --input_mode all --scoring both --output_dir outputs/re_eval_ChtGPT4.1 --batch_size 512
