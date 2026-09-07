#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

LOGGER = logging.getLogger("re_hallu")

CATEGORIES = ["relation_hallucination", "incompleteness", "overgeneration", "context_induced"]
DETERMINISTIC_CATEGORIES = ["relation_hallucination", "incompleteness", "overgeneration"]

NON_DIRECTIONAL_DATASETS = {"BIORED", "DCE"}

DATASET_ROLES = {
    "CHEMPROT": {"arg1", "arg2"},
    "DDI": {"e1", "e2"},
    "DCE": {"span"},
    "BIORED": {"entity1", "entity2"},
}

# Legal labels for a GENERATED relation.  DDI NON is deliberately absent: it
# is a negative-candidate annotation, not something a model predicts.
DATASET_LABELS = {
    "CHEMPROT": ["CPR:3", "CPR:4", "CPR:5", "CPR:6", "CPR:9"],
    "DDI": ["MECHANISM", "EFFECT", "ADVISE", "INT"],
    "DCE": ["POS", "NEG", "COMB"],
    "BIORED": [
        "Association", "Positive_Correlation", "Negative_Correlation", "Bind",
        "Cotreatment", "Comparison", "Drug_Interaction", "Conversion",
    ],
}

NEGATIVE_LABELS = {"DDI": {"NON"}}

# Entity types that participate in relations.  Verified against the corpus:
# BioRED OrganismTaxon (654 mentions) and CellLine (90) never appear in any of
# the 5,935 gold relations, so a generated relation touching one is invalid.
RELATION_ENTITY_TYPES = {
    "CHEMPROT": {"CHEMICAL", "GENE-Y", "GENE-N"},
    "DDI": {"DRUG", "BRAND", "GROUP", "DRUG_N"},
    "DCE": {"DRUG"},
    "BIORED": {
        "GeneOrGeneProduct", "DiseaseOrPhenotypicFeature",
        "ChemicalEntity", "SequenceVariant",
    },
}

ROLE_TYPE_CONSTRAINTS = {"CHEMPROT": {"arg1": {"CHEMICAL"}, "arg2": {"GENE-Y", "GENE-N"}}}

# Semantically adjacent / inverting substitutions.  These make the
# perturbation clinically meaningful rather than arbitrary: CPR:3 -> CPR:4
# turns an upregulator into a downregulator; Positive_Correlation ->
# Negative_Correlation inverts a gene-disease claim; DCE POS -> NEG inverts
# combination efficacy.
ADJACENT_LABELS = {
    "CHEMPROT": {
        "CPR:3": ["CPR:4"], "CPR:4": ["CPR:3"],
        "CPR:5": ["CPR:6"], "CPR:6": ["CPR:5"],
        "CPR:9": ["CPR:4", "CPR:3"],
    },
    "DDI": {
        "MECHANISM": ["EFFECT"], "EFFECT": ["MECHANISM"],
        "ADVISE": ["EFFECT"], "INT": ["EFFECT"],
    },
    "DCE": {"POS": ["NEG"], "NEG": ["POS"], "COMB": ["POS"]},
    "BIORED": {
        "Positive_Correlation": ["Negative_Correlation"],
        "Negative_Correlation": ["Positive_Correlation"],
        "Association": ["Positive_Correlation", "Negative_Correlation"],
        "Bind": ["Association"],
        "Cotreatment": ["Drug_Interaction"],
        "Drug_Interaction": ["Cotreatment"],
        "Comparison": ["Association"],
        "Conversion": ["Association"],
    },
}

LABEL_NAMESPACE = {
    "CHEMPROT": "CHEMPROT_CPR",
    "DDI": "DDI_CLASS",
    "DCE": "DCE_CLASS",
    "BIORED": "BIORED_RELATION",
}

NEGATIVE_POLARITY_LABELS = {"NEG", "Negative_Correlation"}
NONPOLAR_LABELS = {
    "COMB", "Association", "Bind", "Comparison", "Conversion",
    "Cotreatment", "Drug_Interaction",
}

# Keys the DETECTION pipeline must strip before showing a record to a model.
PROMPT_UNSAFE_KEYS = (
    "hallucination_provenance",
    "hallucination_category",
    "is_hallucination_benchmark",
    "source_benchmark_id",
    "metadata",
)


# =====================================================================
# IO
# =====================================================================

def setup_logging(output_dir: str, level: str = "INFO") -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "generation.log"), encoding="utf-8"),
        ],
        force=True,
    )


def load_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Malformed JSON on line %d: %s", lineno, exc)


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def strip_reasoning(text: str) -> str:
    """Remove Qwen3-style reasoning blocks.

    Qwen3 emits <think> ... </think> before the answer when thinking mode is
    on. That block contains braces, so a naive scan for the first '{' can land
    inside the model's scratchpad. Handles an unclosed <think> too, which
    happens when generation is cut off by max_tokens.
    """
    if "<think>" not in text:
        return text
    while "<think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            return text[:start]          # unclosed: everything after is scratchpad
        text = text[:start] + text[end + len("</think>"):]
    return text.strip()


def extract_json_block(raw: str) -> Optional[Dict[str, Any]]:
    """First balanced JSON object in a model response."""
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


# =====================================================================
# Schema helpers
# =====================================================================

def dataset_of(row: Dict[str, Any]) -> str:
    return (row.get("dataset") or "").upper()


def is_directional(row: Dict[str, Any]) -> bool:
    return dataset_of(row) not in NON_DIRECTIONAL_DATASETS


def negative_labels(row: Dict[str, Any]) -> Set[str]:
    return NEGATIVE_LABELS.get(dataset_of(row), set())


def gold_relations(row: Dict[str, Any], include_negative: bool = False) -> List[Dict[str, Any]]:
    neg = negative_labels(row)
    return [r for r in (row.get("relations") or [])
            if include_negative or r.get("label") not in neg]


def negative_relations(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    neg = negative_labels(row)
    return [r for r in (row.get("relations") or []) if r.get("label") in neg]


def legal_labels(row: Dict[str, Any]) -> Set[str]:
    ds = dataset_of(row)
    labels = set(DATASET_LABELS.get(ds, []))
    if not labels:
        labels = {r.get("label") for r in gold_relations(row) if r.get("label")}
    return labels - negative_labels(row)


def legal_roles(row: Dict[str, Any]) -> Set[str]:
    roles = set(DATASET_ROLES.get(dataset_of(row), set()))
    if roles:
        return roles
    return {a.get("role", "") for r in (row.get("relations") or [])
            for a in (r.get("arguments") or []) if a.get("role")}


def entity_lookup(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ent in row.get("entities") or []:
        if "entity_id" in ent:
            out[str(ent["entity_id"])] = ent
    for men in row.get("mentions") or []:
        if "mention_id" in men:
            out.setdefault(str(men["mention_id"]), men)
    return out


def entity_type_of(row: Dict[str, Any], entity_id: str) -> str:
    return (entity_lookup(row).get(str(entity_id)) or {}).get("type", "")


def entity_surface(row: Dict[str, Any], entity_id: str) -> str:
    ent = entity_lookup(row).get(str(entity_id)) or {}
    if ent.get("text"):
        return ent["text"]
    sf = ent.get("surface_forms") or []
    return sf[0] if sf else str(entity_id)


def canonical_argument_tuple(rel: Dict[str, Any], row: Dict[str, Any]) -> Tuple:
    pairs = [(a.get("role", ""), str(a.get("entity_id", "")))
             for a in (rel.get("arguments") or []) if isinstance(a, dict)]
    if not is_directional(row):
        return tuple(sorted(eid for _, eid in pairs))
    return tuple(pairs)


def canonical_relation_signature(rel: Dict[str, Any], row: Dict[str, Any]) -> Tuple:
    return (rel.get("label", ""), canonical_argument_tuple(rel, row))


def gold_signature_set(row: Dict[str, Any]) -> Set[Tuple]:
    return {canonical_relation_signature(r, row) for r in gold_relations(row)}


def gold_argument_tuples(row: Dict[str, Any]) -> Set[Tuple]:
    return {canonical_argument_tuple(r, row) for r in gold_relations(row)}


def unordered_pair_key(rel: Dict[str, Any]) -> Tuple:
    return tuple(sorted(str(a.get("entity_id", "")) for a in (rel.get("arguments") or [])))


def gold_pair_keys(row: Dict[str, Any]) -> Set[Tuple]:
    return {unordered_pair_key(r) for r in gold_relations(row)}


def relation_arg_ids(rel: Dict[str, Any]) -> List[str]:
    return [str(a["entity_id"]) for a in (rel.get("arguments") or []) if isinstance(a, dict)]


def minimal(rel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": rel.get("label", ""),
        "arguments": [{"role": a.get("role", ""), "entity_id": str(a.get("entity_id", ""))}
                      for a in (rel.get("arguments") or []) if isinstance(a, dict)],
    }


def infer_polarity(label: str) -> str:
    if label in NEGATIVE_POLARITY_LABELS:
        return "negative"
    if label in NONPOLAR_LABELS:
        return "nonpolar"
    return "positive"


def type_pair_legal(ds: str, ta: str, tb: str,
                    legal_pairs: Dict[str, Set[Tuple[str, str]]]) -> bool:
    allowed = RELATION_ENTITY_TYPES.get(ds, set())
    if allowed and (ta not in allowed or tb not in allowed):
        return False
    pairs = legal_pairs.get(ds)
    if pairs:
        return tuple(sorted((ta, tb))) in pairs
    return True


# =====================================================================
# Corpus-derived priors
# =====================================================================

@dataclass
class CorpusPriors:
    """Type-pair -> label distributions learned from gold.

    Fabricated relations are labelled by sampling this distribution, so an
    overgenerated ChemicalEntity/DiseaseOrPhenotypicFeature relation gets
    Positive_Correlation far more often than Conversion -- i.e. it looks like
    something a model would plausibly emit.
    """
    label_by_type_pair: Dict[str, Dict[Tuple[str, str], Counter]] = field(default_factory=dict)
    legal_type_pairs: Dict[str, Set[Tuple[str, str]]] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: Sequence[Dict[str, Any]]) -> "CorpusPriors":
        by_pair: Dict[str, Dict[Tuple[str, str], Counter]] = defaultdict(lambda: defaultdict(Counter))
        legal: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
        for row in rows:
            ds = dataset_of(row)
            for rel in gold_relations(row):
                ids = relation_arg_ids(rel)
                types = tuple(sorted(entity_type_of(row, i) for i in ids))
                if len(types) != 2:
                    continue
                by_pair[ds][types][rel.get("label", "")] += 1
                legal[ds].add(types)
        return cls(label_by_type_pair={k: dict(v) for k, v in by_pair.items()},
                   legal_type_pairs={k: set(v) for k, v in legal.items()})

    def sample_label(self, ds: str, ta: str, tb: str, rng: random.Random,
                     exclude: Optional[Set[str]] = None) -> Optional[str]:
        exclude = exclude or set()
        dist = self.label_by_type_pair.get(ds, {}).get(tuple(sorted((ta, tb))))
        pool: List[str] = []
        weights: List[int] = []
        if dist:
            for lab, cnt in dist.items():
                if lab not in exclude:
                    pool.append(lab)
                    weights.append(cnt)
        if not pool:
            pool = [l for l in DATASET_LABELS.get(ds, []) if l not in exclude]
            weights = [1] * len(pool)
        if not pool:
            return None
        return rng.choices(pool, weights=weights, k=1)[0]


# =====================================================================
# Validation
# =====================================================================

@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)


def check_relation_structure(rel: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    ds = dataset_of(row)
    valid_ids = set(entity_lookup(row).keys())
    args = rel.get("arguments")
    if not isinstance(args, list):
        return ["arguments_not_list"]

    if len(args) < 2:
        errors.append(f"too_few_arguments:{len(args)}")
    if ds in {"CHEMPROT", "DDI", "BIORED"} and len(args) != 2:
        errors.append(f"wrong_arity:{len(args)}")

    label = rel.get("label")
    allowed = legal_labels(row)
    if not label:
        errors.append("missing_label")
    elif allowed and label not in allowed:
        errors.append(f"illegal_label:{label}")

    roles_allowed = legal_roles(row)
    type_constraints = ROLE_TYPE_CONSTRAINTS.get(ds, {})
    bearing = RELATION_ENTITY_TYPES.get(ds, set())
    ids: List[str] = []
    for arg in args:
        if not isinstance(arg, dict):
            errors.append("argument_not_dict")
            continue
        eid = str(arg.get("entity_id", ""))
        ids.append(eid)
        if eid not in valid_ids:
            errors.append(f"unknown_entity_id:{eid}")
            continue
        etype = entity_type_of(row, eid)
        if bearing and etype and etype not in bearing:
            errors.append(f"entity_type_not_relation_bearing:{etype}")
        role = arg.get("role")
        if not role:
            errors.append("missing_role")
        elif roles_allowed and role not in roles_allowed:
            errors.append(f"unexpected_role:{role}")
        if role in type_constraints and etype and etype not in type_constraints[role]:
            errors.append(f"role_type_mismatch:{role}:{etype}")
    if len(ids) != len(set(ids)):
        errors.append("duplicate_argument_entity")
    return errors


def validate_instance(relations: Sequence[Dict[str, Any]], row: Dict[str, Any],
                      category: str, new_context: Optional[str] = None) -> ValidationResult:
    """Validate a FULL emitted relation set against the source row.

    Unlike the previous pipeline this constrains the whole set, not just the
    presence of one perturbed relation.
    """
    errors: List[str] = []
    rels = [r for r in relations if isinstance(r, dict)]
    if not rels:
        return ValidationResult(False, ["empty_relations"])

    for rel in rels:
        errors.extend(check_relation_structure(rel, row))

    golds = gold_relations(row)
    gold_sigs = gold_signature_set(row)
    gold_args = gold_argument_tuples(row)
    pred_sigs = {canonical_relation_signature(r, row) for r in rels}

    if category == "relation_hallucination":
        want = sorted(canonical_argument_tuple(r, row) for r in golds)
        got = sorted(canonical_argument_tuple(r, row) for r in rels)
        if want != got:
            errors.append("relation_hallucination_argument_set_changed")
        gold_map = {canonical_argument_tuple(r, row): r.get("label") for r in golds}
        changed = [r for r in rels
                   if canonical_argument_tuple(r, row) in gold_map
                   and r.get("label") != gold_map[canonical_argument_tuple(r, row)]]
        if not changed:
            errors.append("no_relation_label_change")
        for r in changed:
            if canonical_relation_signature(r, row) in gold_sigs:
                errors.append("flip_collides_with_existing_gold_relation")
                break

    elif category == "incompleteness":
        if dataset_of(row) == "DCE" and len(rels) == len(golds):
            gold_by_pos = [set(relation_arg_ids(r)) for r in golds]
            pred_by_pos = [set(relation_arg_ids(r)) for r in rels]
            if not any(p < g for p, g in zip(pred_by_pos, gold_by_pos)):
                errors.append("dce_incompleteness_without_missing_argument")
            if any(len(p) < 2 for p in pred_by_pos):
                errors.append("dce_incompleteness_arity_below_two")
            for p, g in zip(pred_by_pos, gold_by_pos):
                if not p <= g:
                    errors.append("dce_incompleteness_added_argument")
                    break
        else:
            if not pred_sigs < gold_sigs:
                errors.append("incompleteness_not_strict_subset_of_gold")

    elif category in {"overgeneration", "context_induced"}:
        extra = [r for r in rels if canonical_relation_signature(r, row) not in gold_sigs]
        if not extra:
            errors.append(f"{category}_without_new_relation")
        for r in extra:
            if canonical_argument_tuple(r, row) in gold_args:
                errors.append(f"{category}_relabels_existing_positive_relation")
                break
        if golds and (pred_sigs & gold_sigs) != gold_sigs:
            errors.append(f"{category}_dropped_gold_relations")
        if category == "context_induced":
            if new_context is None or not str(new_context).strip():
                errors.append("missing_replacement_context_text")
            elif str(new_context).strip() == (row.get("context_text") or "").strip():
                errors.append("context_not_changed")

    return ValidationResult(ok=not errors, errors=errors)


# =====================================================================
# Deterministic candidate builders
# =====================================================================

@dataclass
class Candidate:
    row: Dict[str, Any]
    category: str
    relations: List[Dict[str, Any]]           # full emitted set, minimal form
    context_text: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


def pick_flip_label(ds: str, current: str, policy: str, rng: random.Random) -> Optional[str]:
    space = [l for l in DATASET_LABELS.get(ds, []) if l != current]
    if not space:
        return None
    if policy == "random":
        return rng.choice(space)
    adjacent = [l for l in ADJACENT_LABELS.get(ds, {}).get(current, []) if l != current]
    if policy == "adjacent":
        return rng.choice(adjacent) if adjacent else rng.choice(space)
    if policy == "distant":
        distant = [l for l in space if l not in adjacent]
        return rng.choice(distant) if distant else rng.choice(space)
    return rng.choice(space)


def build_relation_hallucination(row: Dict[str, Any], flip_k: int, policy: str,
                                 rng: random.Random, max_variants: int) -> List[Candidate]:
    golds = gold_relations(row)
    if not golds:
        return []
    ds = dataset_of(row)
    gold_sigs = gold_signature_set(row)
    out: List[Candidate] = []

    order = list(range(len(golds)))
    rng.shuffle(order)
    for idx in order:
        if len(out) >= max_variants:
            break
        targets = [idx]
        if flip_k > 1:
            others = [j for j in range(len(golds)) if j != idx]
            rng.shuffle(others)
            targets += others[: flip_k - 1]

        rels = [minimal(r) for r in golds]
        flipped: List[Dict[str, Any]] = []
        ok = True
        for t in targets:
            cur = golds[t].get("label", "")
            new = pick_flip_label(ds, cur, policy, rng)
            if new is None:
                ok = False
                break
            cand = dict(rels[t])
            cand["label"] = new
            # 18 ChemProt argument pairs carry two gold labels; flipping onto
            # the sibling label would silently become incompleteness.
            if canonical_relation_signature(cand, row) in gold_sigs:
                ok = False
                break
            rels[t] = cand
            flipped.append({"index": t, "from": cur, "to": new})
        if not ok or not flipped:
            continue
        out.append(Candidate(
            row=row, category="relation_hallucination", relations=rels,
            detail={"flips": flipped, "flip_policy": policy,
                    "corruption_rate": round(len(flipped) / max(1, len(golds)), 4)}))
    return out


def build_incompleteness(row: Dict[str, Any], rng: random.Random,
                         max_variants: int) -> List[Candidate]:
    golds = gold_relations(row)
    ds = dataset_of(row)
    out: List[Candidate] = []

    if len(golds) >= 2:
        order = list(range(len(golds)))
        rng.shuffle(order)
        for idx in order:
            if len(out) >= max_variants:
                break
            rels = [minimal(r) for j, r in enumerate(golds) if j != idx]
            if not rels:
                continue
            out.append(Candidate(
                row=row, category="incompleteness", relations=rels,
                detail={"mode": "drop_relation", "dropped_relation_index": idx,
                        "dropped_label": golds[idx].get("label"),
                        "corruption_rate": round(1 / len(golds), 4)}))

    if ds == "DCE":
        # n-ary argument drop: the combination loses a drug but stays a combination.
        for i, rel in enumerate(golds):
            if len(out) >= max_variants:
                break
            ids = relation_arg_ids(rel)
            if len(ids) < 3:
                continue
            drop_at = rng.randrange(len(ids))
            rels = [minimal(r) for r in golds]
            args = [a for k, a in enumerate(rels[i]["arguments"]) if k != drop_at]
            rels[i] = {"label": rels[i]["label"], "arguments": args}
            out.append(Candidate(
                row=row, category="incompleteness", relations=rels,
                detail={"mode": "drop_argument", "relation_index": i,
                        "dropped_entity_id": ids[drop_at],
                        "corruption_rate": round(1 / len(ids), 4)}))
    return out[:max_variants]


def enumerate_free_pairs(row: Dict[str, Any], priors: CorpusPriors,
                         prefer_annotated_negatives: bool) -> List[Tuple[str, str, bool]]:
    """Entity pairs carrying no positive gold relation.

    The boolean marks pairs the corpus itself annotates as NON -- human-verified
    negatives, the strongest available evidence that a fabricated relation there
    is genuinely unsupported.
    """
    ds = dataset_of(row)
    ents = row.get("entities") or []
    types = {str(e.get("entity_id", "")): e.get("type", "") for e in ents}
    used = gold_pair_keys(row)
    annotated_neg = {unordered_pair_key(r) for r in negative_relations(row)}

    free: List[Tuple[str, str, bool]] = []
    for a, b in combinations([str(e.get("entity_id", "")) for e in ents], 2):
        key = tuple(sorted((a, b)))
        if key in used:
            continue
        if not type_pair_legal(ds, types.get(a, ""), types.get(b, ""), priors.legal_type_pairs):
            continue
        free.append((a, b, key in annotated_neg))
    if prefer_annotated_negatives:
        free.sort(key=lambda t: (not t[2],))
    return free


def build_overgeneration(row: Dict[str, Any], priors: CorpusPriors, rng: random.Random,
                         max_variants: int, prefer_annotated_negatives: bool) -> List[Candidate]:
    ds = dataset_of(row)
    golds = gold_relations(row)
    base = [minimal(r) for r in golds]
    roles = sorted(legal_roles(row))
    out: List[Candidate] = []

    # DCE: append an unused drug to an existing combination.
    if ds == "DCE" and golds:
        used_ids = {i for r in golds for i in relation_arg_ids(r)}
        spare = [str(e["entity_id"]) for e in (row.get("entities") or [])
                 if str(e.get("entity_id", "")) not in used_ids and e.get("type") == "DRUG"]
        rng.shuffle(spare)
        for eid in spare[:max_variants]:
            rels = [dict(r) for r in base]
            rels[0] = {"label": rels[0]["label"],
                       "arguments": list(rels[0]["arguments"]) + [{"role": "span", "entity_id": eid}]}
            out.append(Candidate(
                row=row, category="overgeneration", relations=rels,
                detail={"mode": "append_argument", "added_entity_id": eid,
                        "from_annotated_negative": False}))
        if out:
            return out[:max_variants]

    free = enumerate_free_pairs(row, priors, prefer_annotated_negatives)
    if not prefer_annotated_negatives:
        rng.shuffle(free)
    for a, b, is_neg in free:
        if len(out) >= max_variants:
            break
        ta, tb = entity_type_of(row, a), entity_type_of(row, b)
        label = priors.sample_label(ds, ta, tb, rng)
        if label is None:
            continue
        if ds == "CHEMPROT":
            if ta == "CHEMICAL":
                args = [{"role": "arg1", "entity_id": a}, {"role": "arg2", "entity_id": b}]
            elif tb == "CHEMICAL":
                args = [{"role": "arg1", "entity_id": b}, {"role": "arg2", "entity_id": a}]
            else:
                continue
        elif ds == "DCE":
            args = [{"role": "span", "entity_id": a}, {"role": "span", "entity_id": b}]
        else:
            r1 = roles[0] if roles else "entity1"
            r2 = roles[1] if len(roles) > 1 else "entity2"
            args = [{"role": r1, "entity_id": a}, {"role": r2, "entity_id": b}]

        out.append(Candidate(
            row=row, category="overgeneration",
            relations=base + [{"label": label, "arguments": args}],
            detail={"mode": "add_relation", "added_label": label,
                    "added_entities": [a, b], "added_types": [ta, tb],
                    "from_annotated_negative": is_neg,
                    "corruption_rate": round(1 / (len(base) + 1), 4)}))
    return out


# =====================================================================
# Backends (context_induced only)
# =====================================================================

@dataclass
class GenConfig:
    max_new_tokens: int = 1024
    max_model_len: int = 8192
    max_text_chars: int = 4000
    max_context_chars: int = 2000
    max_entities: int = 80
    max_relations: int = 80
    temperature_schedule: Tuple[float, ...] = (0.0, 0.4, 0.8)
    top_p: float = 0.95
    seed: int = 42


def render_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Apply a chat template with reasoning disabled where supported.

    Qwen3 defaults to thinking mode, which wraps the answer in <think> ...
    </think>. enable_thinking=False turns it off at the template level.
    Tokenizers whose template does not accept the flag raise TypeError, so we
    fall back to the plain call.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


class BaseBackend:
    def __init__(self, system_prompt: str, cfg: GenConfig):
        self.system_prompt = system_prompt
        self.cfg = cfg

    def render_chat(self, user_prompt: str) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def generate_batch(self, prompts: Sequence[str], temperature: float, seed: int) -> List[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class VLLMBackend(BaseBackend):
    def __init__(self, model_path: str, system_prompt: str, cfg: GenConfig,
                 dtype: str = "auto", tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90, enforce_eager: bool = True):
        super().__init__(system_prompt, cfg)
        from vllm import LLM
        from transformers import AutoTokenizer

        LOGGER.info("Loading tokenizer %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        LOGGER.info("Loading vLLM engine (tp=%d, max_model_len=%d, enforce_eager=%s)",
                    tensor_parallel_size, cfg.max_model_len, enforce_eager)
        self.llm = LLM(model=model_path, dtype=dtype,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=cfg.max_model_len,
                       trust_remote_code=True, seed=cfg.seed,
                       enforce_eager=enforce_eager)

    def render_chat(self, user_prompt: str) -> str:
        return render_chat_template(self.tokenizer, self.system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate_batch(self, prompts: Sequence[str], temperature: float, seed: int) -> List[str]:
        from vllm import SamplingParams
        rendered = [self.render_chat(p) for p in prompts]
        kwargs: Dict[str, Any] = {"temperature": float(temperature),
                                  "max_tokens": self.cfg.max_new_tokens, "n": 1}
        if temperature > 0:
            kwargs["top_p"] = self.cfg.top_p
            kwargs["seed"] = int(seed)
        try:
            params = SamplingParams(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            params = SamplingParams(**kwargs)
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


class HFBackend(BaseBackend):
    def __init__(self, model_path: str, system_prompt: str, cfg: GenConfig,
                 dtype: str = "float16"):
        super().__init__(system_prompt, cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dmap = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32, "auto": "auto"}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dmap.get(dtype, torch.float16),
            device_map="auto", trust_remote_code=True)
        self.model.eval()

    def render_chat(self, user_prompt: str) -> str:
        return render_chat_template(self.tokenizer, self.system_prompt, user_prompt)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate_batch(self, prompts: Sequence[str], temperature: float, seed: int) -> List[str]:
        torch = self.torch
        torch.manual_seed(seed)
        out: List[str] = []
        for p in prompts:
            inputs = self.tokenizer(self.render_chat(p), return_tensors="pt").to(self.model.device)
            kw: Dict[str, Any] = {"max_new_tokens": self.cfg.max_new_tokens,
                                  "do_sample": temperature > 0,
                                  "pad_token_id": self.tokenizer.eos_token_id}
            if temperature > 0:
                kw["temperature"] = temperature
                kw["top_p"] = self.cfg.top_p
            with torch.no_grad():
                gen = self.model.generate(**inputs, **kw)
            out.append(self.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True).strip())
        return out


class EchoBackend(BaseBackend):
    def __init__(self, system_prompt: str, cfg: GenConfig,
                 responses: Optional[List[str]] = None):
        super().__init__(system_prompt, cfg)
        self.responses = responses or []

    def render_chat(self, user_prompt: str) -> str:
        return user_prompt

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate_batch(self, prompts: Sequence[str], temperature: float, seed: int) -> List[str]:
        if not self.responses:
            return ["{}"] * len(prompts)
        return [self.responses[i % len(self.responses)] for i in range(len(prompts))]


# =====================================================================
# context_induced prompting
# =====================================================================

def real_context_for(row: Dict[str, Any],
                     sibling_index: Dict[Any, List[Dict[str, Any]]]) -> Optional[str]:
    """Genuine context drawn from the corpus rather than model-generated.

    DDI      : sibling sentences of the same source document (3,724/3,876 rows).
    BIORED   : the title passage (every row carries title + body).
    DCE      : the real paragraph already in context_text.
    CHEMPROT : no internal structure exposed -> None.
    """
    ds = dataset_of(row)
    if ds == "DCE":
        return row.get("context_text") or None
    if ds == "BIORED":
        passages = row.get("passages") or []
        if len(passages) >= 2:
            return (passages[0].get("text") or "").strip() or None
        return None
    if ds == "DDI":
        sibs = sibling_index.get(row.get("parent_doc_id"), [])
        others = [s.get("text", "") for s in sibs
                  if s.get("benchmark_id") != row.get("benchmark_id")]
        joined = " ".join(t.strip() for t in others if t.strip())
        return joined or None
    return None


def build_context_prompt(row: Dict[str, Any], template: str, cfg: GenConfig,
                         free_pairs: Sequence[Tuple[str, str, bool]],
                         previous_errors: Sequence[str] = ()) -> str:
    golds = gold_relations(row)
    ents = [{"entity_id": str(e.get("entity_id", "")), "type": e.get("type", ""),
             "text": e.get("text") or (e.get("surface_forms") or [""])[0]}
            for e in (row.get("entities") or [])[: cfg.max_entities]]
    suggestions = [{"entity_id_1": a, "entity_id_2": b,
                    "text_1": entity_surface(row, a), "text_2": entity_surface(row, b)}
                   for a, b, _ in free_pairs[:12]]
    payload = {
        "dataset": dataset_of(row),
        "annotation_level": row.get("annotation_level", ""),
        "task_type": row.get("task_type", ""),
        "text_unit": row.get("text_unit", ""),
        "target_hallucination_category": "context_induced",
        "text": truncate(row.get("text", ""), cfg.max_text_chars),
        "current_context_text": truncate(row.get("context_text", ""), cfg.max_context_chars),
        "entities": ents,
        "gold_relations": [
            {"label": r.get("label", ""),
             "arguments": [{"role": a.get("role", ""), "entity_id": str(a.get("entity_id", ""))}
                           for a in (r.get("arguments") or [])]}
            for r in golds[: cfg.max_relations]],
        "allowed_relation_labels": sorted(legal_labels(row)),
        "allowed_argument_roles": sorted(legal_roles(row)),
        "unused_entity_pairs": suggestions,
    }
    parts = [
        template,
        ("TASK: context-induced hallucination.\n"
         "1. Choose ONE argument pair from unused_entity_pairs and assert a relation over it "
         "using an allowed label. That relation must NOT be supported by TEXT.\n"
         "2. Write a replacement context_text: a short, natural biomedical passage that makes "
         "the relation seem plausible to a careful reader, without stating it outright and "
         "without contradicting TEXT.\n"
         "3. Do NOT repeat the gold relations -- the pipeline re-attaches them.\n"
         "Write in the register of a biomedical abstract. Do not use meta-language such as "
         "'This row', 'The text suggests', or 'This passage'."),
        'Return ONLY JSON: {"hallucination_category": "context_induced", '
        '"relations": [{"label": str, "arguments": [{"role": str, "entity_id": str}, ...]}], '
        '"context_text": str}',
    ]
    if previous_errors:
        uniq = list(dict.fromkeys(previous_errors))[:8]
        parts.append("Your previous attempt was rejected. Fix all of:\n- " + "\n- ".join(uniq))
    parts.append("INPUT INSTANCE:\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    return "\n\n".join(p for p in parts if p)


# =====================================================================
# Emission
# =====================================================================

def enrich(relations: Sequence[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    ns = LABEL_NAMESPACE.get(dataset_of(row), "RELATION_LABEL")
    out = []
    for i, rel in enumerate(relations):
        args = [{"role": a.get("role", ""), "entity_id": str(a.get("entity_id", ""))}
                for a in (rel.get("arguments") or [])]
        label = rel.get("label", "")
        out.append({"relation_id": f"H{i}", "label": label, "label_namespace": ns,
                    "polarity": infer_polarity(label), "arguments": args,
                    "argument_types": [entity_type_of(row, a["entity_id"]) for a in args],
                    "entity_count": len(args)})
    return out


def recompute_metadata(source_metadata: Dict[str, Any],
                       relations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keep = {"original_document_id", "original_sentence_id", "passage_count",
            "entity_type_inventory", "mention_type_inventory",
            "context_text_source", "context_text_grounding"}
    meta = {k: v for k, v in (source_metadata or {}).items() if k in keep}
    meta["relation_label_inventory"] = dict(Counter(r.get("label", "") for r in relations))
    meta["relation_polarity_inventory"] = dict(Counter(r.get("polarity", "") for r in relations))
    meta["relation_arity_inventory"] = dict(
        Counter(str(len(r.get("arguments") or [])) for r in relations))
    return meta


def emit_row(cand: Candidate, variant_index: int, args: argparse.Namespace,
             generator: str, extra_provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    src = cand.row
    row = copy.deepcopy(src)
    base_id = src.get("benchmark_id", "row")
    suffix = "" if variant_index == 0 else f"_{variant_index}"
    row["benchmark_id"] = f"{base_id}_HAL_{cand.category}{suffix}"
    row["source_benchmark_id"] = base_id
    row["hallucination_category"] = cand.category
    row["is_hallucination_benchmark"] = True

    rels = list(cand.relations)
    if args.keep_negative_relations and cand.category in {"overgeneration", "context_induced"}:
        rels = rels + [minimal(r) for r in negative_relations(src)]

    row["relations"] = enrich(rels, src)
    row["relation_count"] = len(row["relations"])
    row["entity_count"] = len(src.get("entities") or src.get("mentions") or [])
    if cand.context_text is not None:
        row["context_text"] = cand.context_text
    row["metadata"] = recompute_metadata(src.get("metadata") or {}, row["relations"])

    prov = {
        "construction": "deterministic" if cand.category in DETERMINISTIC_CATEGORIES else "llm",
        "generator": generator,
        "category": cand.category,
        "source_positive_relation_count": len(gold_relations(src)),
        "source_negative_relation_count": len(negative_relations(src)),
        "emitted_relation_count": len(row["relations"]),
        "variant_index": variant_index,
        "negatives_retained": bool(args.keep_negative_relations),
        "seed": args.seed,
    }
    prov.update(cand.detail or {})
    if extra_provenance:
        prov.update(extra_provenance)
    row["hallucination_provenance"] = prov
    return row


def strip_prompt_unsafe(row: Dict[str, Any]) -> Dict[str, Any]:
    """For the DETECTION pipeline: remove everything that leaks the answer."""
    clean = copy.deepcopy(row)
    for key in PROMPT_UNSAFE_KEYS:
        clean.pop(key, None)
    return clean


# =====================================================================
# Quota sampling
# =====================================================================

def select_to_target(candidates_by_row: List[List[Candidate]], target: Optional[int],
                     rng: random.Random) -> List[Tuple[Candidate, int]]:
    """Round-robin across source rows.

    Every eligible row contributes its first variant before any row contributes
    a second, so source coverage is maximised and the dataset mix stays
    proportional to eligibility.
    """
    buckets = [list(c) for c in candidates_by_row if c]
    rng.shuffle(buckets)
    for b in buckets:
        rng.shuffle(b)

    selected: List[Tuple[Candidate, int]] = []
    depth = 0
    while buckets:
        progressed = False
        for bucket in buckets:
            if depth < len(bucket):
                selected.append((bucket[depth], depth))
                progressed = True
                if target is not None and len(selected) >= target:
                    return selected
        if not progressed:
            break
        depth += 1
    return selected


# =====================================================================
# Driver
# =====================================================================

def run(args: argparse.Namespace) -> int:
    setup_logging(args.output_dir, args.log_level)
    rng = random.Random(args.seed)

    rows = list(load_jsonl(args.input_jsonl))
    if args.datasets:
        keep = {d.strip().upper() for d in args.datasets.split(",") if d.strip()}
        rows = [r for r in rows if dataset_of(r) in keep]
    if args.max_rows:
        rows = rows[: args.max_rows]
    LOGGER.info("Loaded %d source rows.", len(rows))
    if not rows:
        LOGGER.error("No rows loaded.")
        return 1

    priors = CorpusPriors.fit(rows)
    LOGGER.info("Learned type-pair label priors for %d datasets.", len(priors.label_by_type_pair))

    sibling_index: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if dataset_of(r) == "DDI":
            sibling_index[r.get("parent_doc_id")].append(r)

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    targets = {
        "relation_hallucination": args.target_relation_hallucination,
        "incompleteness": args.target_incompleteness,
        "overgeneration": args.target_overgeneration,
        "context_induced": args.target_context_induced,
    }

    out_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, Any]] = {}

    # ---------------- deterministic categories ----------------
    for category in [c for c in categories if c in DETERMINISTIC_CATEGORIES]:
        LOGGER.info("Enumerating candidates: %s", category)
        by_row: List[List[Candidate]] = []
        for row in rows:
            if category == "relation_hallucination":
                cands = build_relation_hallucination(
                    row, args.flip_k, args.flip_policy, rng, args.max_variants_per_row)
            elif category == "incompleteness":
                cands = build_incompleteness(row, rng, args.max_variants_per_row)
            else:
                cands = build_overgeneration(row, priors, rng, args.max_variants_per_row,
                                             args.prefer_annotated_negatives)
            cands = [c for c in cands if validate_instance(c.relations, row, category).ok]
            if cands:
                by_row.append(cands)

        total = sum(len(b) for b in by_row)
        target = targets.get(category)
        chosen = select_to_target(by_row, target, rng)
        LOGGER.info("%s: %d eligible rows, %d candidates, %d selected (target=%s)",
                    category, len(by_row), total, len(chosen), target)
        if target and len(chosen) < target:
            LOGGER.warning("%s: only %d constructible, short of target %d.",
                           category, len(chosen), target)
        for cand, vidx in chosen:
            out_rows.append(emit_row(cand, vidx, args, generator="deterministic"))
        stats[category] = {"eligible_rows": len(by_row), "candidates_enumerated": total,
                           "accepted": len(chosen), "target": target,
                           "construction": "deterministic"}

    # ---------------- context_induced (LLM) ----------------
    if "context_induced" in categories and args.backend != "none":
        cfg = GenConfig(
            max_new_tokens=args.max_new_tokens, max_model_len=args.max_model_len,
            max_text_chars=args.max_text_chars, max_context_chars=args.max_context_chars,
            max_entities=args.max_entities, max_relations=args.max_relations,
            temperature_schedule=tuple(float(x) for x in args.temperatures.split(",") if x.strip()),
            seed=args.seed)

        with open(os.path.join(args.prompt_dir, "system.txt"), "r", encoding="utf-8") as fh:
            system_prompt = fh.read().strip()
        templates = {}
        for name in ["CHEMPROT", "DDI", "DCE", "BIORED"]:
            p = os.path.join(args.prompt_dir, f"{name}.txt")
            templates[name] = open(p, encoding="utf-8").read().strip() if os.path.exists(p) else ""

        if args.backend == "vllm":
            backend: BaseBackend = VLLMBackend(
                args.model_path, system_prompt, cfg, dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager)
        elif args.backend == "hf":
            backend = HFBackend(args.model_path, system_prompt, cfg, dtype=args.dtype)
        else:
            backend = EchoBackend(system_prompt, cfg)

        target = targets["context_induced"] or 0
        eligible: List[Tuple[Dict[str, Any], List[Tuple[str, str, bool]], Optional[str]]] = []
        skipped: Counter = Counter()
        for row in rows:
            free = enumerate_free_pairs(row, priors, args.prefer_annotated_negatives)
            if not free:
                skipped["no_free_entity_pair"] += 1
                continue
            ctx = row.get("context_text")
            if args.context_source == "real":
                real = real_context_for(row, sibling_index)
                if not real:
                    skipped["no_real_context_available"] += 1
                    continue
                ctx = real
            if not (ctx or "").strip():
                skipped["no_context_text"] += 1
                continue
            eligible.append((row, free, ctx))
        rng.shuffle(eligible)
        LOGGER.info("context_induced: %d eligible rows (skipped %s)", len(eligible), dict(skipped))

        accepted: List[Tuple[Candidate, Dict[str, Any]]] = []
        pending = eligible[: int(target * args.oversample_factor)] if target else eligible
        errors_by_row: Dict[str, List[str]] = {}
        error_counter: Counter = Counter()
        oversized = 0

        for rnd, temperature in enumerate(cfg.temperature_schedule):
            if (target and len(accepted) >= target) or not pending:
                break
            LOGGER.info("context_induced round %d: %d pending (T=%.2f)",
                        rnd + 1, len(pending), temperature)
            prompts: List[str] = []
            runnable: List[Tuple[Dict[str, Any], List[Tuple[str, str, bool]], Optional[str]]] = []
            for row, free, ctx in pending:
                staged = dict(row)
                staged["context_text"] = ctx
                prompt = build_context_prompt(
                    staged, templates.get(dataset_of(row), ""), cfg, free,
                    errors_by_row.get(row.get("benchmark_id", ""), []))
                n = backend.count_tokens(backend.render_chat(prompt))
                if n > cfg.max_model_len - cfg.max_new_tokens:
                    oversized += 1
                    error_counter["prompt_exceeds_context_window"] += 1
                    continue
                prompts.append(prompt)
                runnable.append((row, free, ctx))
            if not runnable:
                break

            raws: List[str] = []
            for s in range(0, len(prompts), args.batch_size):
                raws.extend(backend.generate_batch(prompts[s:s + args.batch_size],
                                                   temperature, cfg.seed + rnd))
                LOGGER.info("  %d/%d generated",
                            min(s + args.batch_size, len(prompts)), len(prompts))

            still: List[Tuple[Dict[str, Any], List[Tuple[str, str, bool]], Optional[str]]] = []
            for (row, free, ctx), raw in zip(runnable, raws):
                staged = dict(row)
                staged["context_text"] = ctx
                pred = extract_json_block(raw)
                if not pred:
                    error_counter["json_parse_failure"] += 1
                    errors_by_row[row.get("benchmark_id", "")] = ["json_parse_failure"]
                    still.append((row, free, ctx))
                    continue
                extra = [minimal(r) for r in (pred.get("relations") or []) if isinstance(r, dict)]
                full = [minimal(r) for r in gold_relations(row)] + extra
                new_ctx = pred.get("context_text")
                res = validate_instance(full, staged, "context_induced", new_context=new_ctx)
                if res.ok:
                    accepted.append((Candidate(
                        row=row, category="context_induced", relations=full,
                        context_text=str(new_ctx).strip(),
                        detail={"context_source": args.context_source,
                                "original_context_replaced": True,
                                "rounds_used": rnd + 1,
                                "added_relation_count": len(extra),
                                "corruption_rate": round(len(extra) / max(1, len(full)), 4)}),
                        {"temperature": temperature}))
                    if target and len(accepted) >= target:
                        break
                else:
                    for e in res.errors:
                        error_counter[e.split(":")[0]] += 1
                    errors_by_row[row.get("benchmark_id", "")] = res.errors
                    still.append((row, free, ctx))
            pending = still

        final = accepted[:target] if target else accepted
        for cand, prov in final:
            out_rows.append(emit_row(cand, 0, args, generator=args.model_path,
                                     extra_provenance=prov))
        backend.close()

        stats["context_induced"] = {
            "eligible_rows": len(eligible),
            "attempted": len(eligible[: int(target * args.oversample_factor)] if target else eligible),
            "accepted": len(final), "target": target, "construction": "llm",
            "prompts_exceeding_window": oversized,
            "rejection_reasons": dict(error_counter.most_common()),
            "skipped_ineligible": dict(skipped)}

    # ---------------- summary ----------------
    covered = {r["source_benchmark_id"] for r in out_rows}
    failed = [r.get("benchmark_id") for r in rows if r.get("benchmark_id") not in covered]

    per_cat_dataset: Dict[str, Counter] = defaultdict(Counter)
    for r in out_rows:
        per_cat_dataset[r["hallucination_category"]][dataset_of(r)] += 1

    summary = {
        "config": {
            "input_jsonl": args.input_jsonl, "categories": categories,
            "backend": args.backend, "model_path": args.model_path,
            "flip_policy": args.flip_policy, "flip_k": args.flip_k,
            "context_source": args.context_source,
            "prefer_annotated_negatives": args.prefer_annotated_negatives,
            "keep_negative_relations": args.keep_negative_relations,
            "max_variants_per_row": args.max_variants_per_row, "seed": args.seed,
        },
        "source_rows": len(rows),
        "produced_rows": len(out_rows),
        "failed_source_rows": len(failed),
        "categories": {c: {"accepted": s["accepted"], "target": s.get("target"),
                           "construction": s["construction"],
                           "eligible_rows": s.get("eligible_rows"),
                           "candidates_enumerated": s.get("candidates_enumerated")}
                       for c, s in stats.items()},
        "category_detail": stats,
        "per_dataset": dict(Counter(dataset_of(r) for r in out_rows)),
        "per_category_per_dataset": {k: dict(v) for k, v in per_cat_dataset.items()},
        "failed_source_ids": failed[:5000],
    }

    bench_path = os.path.join(args.output_dir, "re_hallu_benchmark.jsonl")
    summ_path = os.path.join(args.output_dir, "re_hallu_summary.json")
    write_jsonl(bench_path, out_rows)
    with open(summ_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    LOGGER.info("=" * 68)
    LOGGER.info("%-26s %10s %10s %14s", "Category", "Accepted", "Target", "Construction")
    for c in categories:
        if c in stats:
            s = stats[c]
            LOGGER.info("%-26s %10d %10s %14s", c, s["accepted"], s.get("target"),
                        s["construction"])
    LOGGER.info("%-26s %10d", "Total Generated", len(out_rows))
    LOGGER.info("%-26s %10d", "Source Rows", len(rows))
    LOGGER.info("%-26s %10d", "Failed Sources", len(failed))
    LOGGER.info("=" * 68)
    LOGGER.info("Benchmark: %s", bench_path)
    LOGGER.info("Summary:   %s", summ_path)
    return 0


# =====================================================================
# Self-test
# =====================================================================

def self_test() -> int:
    row = {
        "benchmark_id": "T1", "dataset": "BIORED", "text": "t", "context_text": "c",
        "entities": [
            {"entity_id": "V1", "type": "SequenceVariant", "surface_forms": ["v1"]},
            {"entity_id": "D1", "type": "DiseaseOrPhenotypicFeature", "surface_forms": ["d1"]},
            {"entity_id": "D2", "type": "DiseaseOrPhenotypicFeature", "surface_forms": ["d2"]},
            {"entity_id": "G1", "type": "GeneOrGeneProduct", "surface_forms": ["g1"]},
            {"entity_id": "TAX", "type": "OrganismTaxon", "surface_forms": ["human"]},
        ],
        "relations": [
            {"relation_id": "R0", "label": "Positive_Correlation",
             "arguments": [{"role": "entity1", "entity_id": "V1"},
                           {"role": "entity2", "entity_id": "D1"}]},
            {"relation_id": "R1", "label": "Association",
             "arguments": [{"role": "entity1", "entity_id": "G1"},
                           {"role": "entity2", "entity_id": "D1"}]},
        ],
    }
    ddi = {
        "benchmark_id": "T2", "dataset": "DDI", "text": "t", "context_text": "c",
        "entities": [{"entity_id": "e0", "text": "a", "type": "BRAND"},
                     {"entity_id": "e1", "text": "b", "type": "DRUG"},
                     {"entity_id": "e2", "text": "c", "type": "DRUG"},
                     {"entity_id": "e3", "text": "d", "type": "DRUG"}],
        "relations": [
            {"relation_id": "p0", "label": "NON",
             "arguments": [{"role": "e1", "entity_id": "e1"}, {"role": "e2", "entity_id": "e2"}]},
            {"relation_id": "p1", "label": "EFFECT",
             "arguments": [{"role": "e1", "entity_id": "e0"}, {"role": "e2", "entity_id": "e1"}]},
            # makes DRUG-DRUG an observed legal type pair, as it is corpus-wide
            {"relation_id": "p2", "label": "MECHANISM",
             "arguments": [{"role": "e1", "entity_id": "e1"}, {"role": "e2", "entity_id": "e3"}]},
        ],
    }

    def R(lab, a, b, r1="entity1", r2="entity2"):
        return {"label": lab, "arguments": [{"role": r1, "entity_id": a},
                                            {"role": r2, "entity_id": b}]}

    cases = [
        ("rel_hallu drops relations", [R("Negative_Correlation", "V1", "D1")],
         row, "relation_hallucination", None, False),
        ("rel_hallu preserves set",
         [R("Negative_Correlation", "V1", "D1"), R("Association", "G1", "D1")],
         row, "relation_hallucination", None, True),
        ("incompleteness with mutated survivor", [R("Negative_Correlation", "V1", "D2")],
         row, "incompleteness", None, False),
        ("incompleteness exact subset", [R("Positive_Correlation", "V1", "D1")],
         row, "incompleteness", None, True),
        ("overgen relabels gold pair",
         [R("Positive_Correlation", "V1", "D1"), R("Association", "G1", "D1"),
          R("Bind", "V1", "D1")], row, "overgeneration", None, False),
        ("overgen new pair",
         [R("Positive_Correlation", "V1", "D1"), R("Association", "G1", "D1"),
          R("Association", "G1", "D2")], row, "overgeneration", None, True),
        ("overgen drops gold", [R("Association", "G1", "D2")],
         row, "overgeneration", None, False),
        ("relation on OrganismTaxon",
         [R("Positive_Correlation", "V1", "D1"), R("Association", "G1", "D1"),
          R("Association", "TAX", "D2")], row, "overgeneration", None, False),
        ("context_induced unchanged context",
         [R("Positive_Correlation", "V1", "D1"), R("Association", "G1", "D1"),
          R("Association", "G1", "D2")], row, "context_induced", "c", False),
        ("context_induced valid",
         [R("Positive_Correlation", "V1", "D1"), R("Association", "G1", "D1"),
          R("Association", "G1", "D2")], row, "context_induced", "rewritten passage", True),
        ("DDI overgen on annotated NON pair",
         [R("EFFECT", "e0", "e1", "e1", "e2"), R("MECHANISM", "e1", "e3", "e1", "e2"),
          R("MECHANISM", "e1", "e2", "e1", "e2")],
         ddi, "overgeneration", None, True),
        ("DDI generated NON label illegal",
         [R("EFFECT", "e0", "e1", "e1", "e2"), R("MECHANISM", "e1", "e3", "e1", "e2"),
          R("NON", "e0", "e2", "e1", "e2")],
         ddi, "overgeneration", None, False),
    ]

    failures = 0
    for name, rels, r, cat, ctx, expect in cases:
        res = validate_instance(rels, r, cat, new_context=ctx)
        status = "PASS" if res.ok == expect else "FAIL"
        failures += status == "FAIL"
        print(f"[{status}] {name}: ok={res.ok} expected={expect} errors={res.errors}")

    rng = random.Random(0)
    c1 = build_relation_hallucination(row, 1, "adjacent", rng, 8)
    ok = bool(c1) and all(len(c.relations) == 2 for c in c1)
    print(f"[{'PASS' if ok else 'FAIL'}] builder: rel_hallu keeps all gold relations "
          f"({len(c1)} variants)")
    failures += not ok

    c2 = build_incompleteness(row, rng, 8)
    ok = bool(c2) and all(len(c.relations) == 1 for c in c2)
    print(f"[{'PASS' if ok else 'FAIL'}] builder: incompleteness emits proper subsets "
          f"({len(c2)} variants)")
    failures += not ok

    priors = CorpusPriors.fit([row, ddi])
    c3 = build_overgeneration(ddi, priors, rng, 4, True)
    ok = bool(c3) and c3[0].detail.get("from_annotated_negative") is True
    print(f"[{'PASS' if ok else 'FAIL'}] builder: DDI overgeneration prefers annotated NON pairs")
    failures += not ok

    total = len(cases) + 3
    print(f"\n{total - failures}/{total} checks passed.")
    return 1 if failures else 0


# =====================================================================
# CLI
# =====================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the HalluRE-Med hallucination benchmark.")
    p.add_argument("--input_jsonl", type=str, default="")
    p.add_argument("--output_dir", type=str, default="outputs/re_hallu")
    p.add_argument("--prompt_dir", type=str, default="prompts_re_hallu")
    p.add_argument("--model_path", type=str, default="", help="Generator for context_induced.")

    p.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf", "echo", "none"],
                   help="'none' skips context_induced and builds only deterministic categories.")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--enforce_eager", action="store_true", default=True,
                   help="Skip torch.compile/CUDA-graph capture. Default True: this avoids "
                        "a Triton JIT step that shells out to gcc, which is fragile on "
                        "compute nodes missing Python.h or a discoverable libcuda.so.1, "
                        "and torch.compile has limited benefit on pre-Ampere GPUs (e.g. "
                        "V100, compute capability 7.0) anyway.")
    p.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false",
                   help="Enable torch.compile/CUDA-graphs. Try this only after confirming "
                        "gcc can build a CUDA extension on this node (see troubleshooting).")

    p.add_argument("--categories", type=str, default=",".join(CATEGORIES))
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--max_rows", type=int, default=None)

    p.add_argument("--target_relation_hallucination", type=int, default=5281)
    p.add_argument("--target_incompleteness", type=int, default=3087)
    p.add_argument("--target_overgeneration", type=int, default=4815)
    p.add_argument("--target_context_induced", type=int, default=4470)
    p.add_argument("--max_variants_per_row", type=int, default=4,
                   help="Cap on instances of one category from one source row.")

    p.add_argument("--flip_policy", type=str, default="adjacent",
                   choices=["adjacent", "random", "distant"])
    p.add_argument("--flip_k", type=int, default=1)
    p.add_argument("--prefer_annotated_negatives", action="store_true", default=True)
    p.add_argument("--no_prefer_annotated_negatives", dest="prefer_annotated_negatives",
                   action="store_false")
    p.add_argument("--keep_negative_relations", action="store_true",
                   help="Retain DDI NON pairs in emitted records (not recommended).")
    p.add_argument("--context_source", type=str, default="existing",
                   choices=["existing", "real"],
                   help="'existing' uses context_text as shipped (Qwen-generated for "
                        "CHEMPROT/DDI/BIORED); 'real' uses DDI sibling sentences, the BioRED "
                        "title passage, and the DCE paragraph.")

    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_text_chars", type=int, default=4000)
    p.add_argument("--max_context_chars", type=int, default=2000)
    p.add_argument("--max_entities", type=int, default=80)
    p.add_argument("--max_relations", type=int, default=80)
    p.add_argument("--temperatures", type=str, default="0.0,0.4,0.8")
    p.add_argument("--oversample_factor", type=float, default=1.6,
                   help="Rows attempted for context_induced as a multiple of the target.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--self_test", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.self_test:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return self_test()
    if not args.input_jsonl:
        print("--input_jsonl is required (or use --self_test).", file=sys.stderr)
        return 2
    if args.backend in {"vllm", "hf"} and not args.model_path:
        print("--model_path is required for context_induced; use --backend none to skip it.",
              file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

#python run_hallu_detection.py --hallu_jsonl data/re_hallu_full/re_hallu_benchmark.jsonl \
#  --unified_jsonl data/unified_RE_benchmark.jsonl \
#  --model_path /path/Qwen3-14B --backend vllm --tensor_parallel_size 2