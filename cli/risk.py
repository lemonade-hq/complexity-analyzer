"""Risk assessment, scored independently of complexity (sun-security fork, PLT-4017).

Risk used to be the fifth complexity dimension, scored in the same call under
a prompt that frames everything as developer effort. Observed: risk tracked
complexity at r=0.91-0.93 and a one-line prod KEDA or ArgoCD upgrade labeled
risk:6-17. Risk is now its own call on three factors the caller multiplies:

    risk_raw = blast_radius * severity * hazard      (each 1-5, raw 1-125)

A multiplicative model keeps a hazardous change confined to dev low and a
no-op in prod low, and only rates a change high when it is wide, severe and
likely to misbehave at once.

What the LLM cannot know from a diff — that a path is a cluster-wide
controller in prod, or that a Terraform module fans out to every account — is
recorded per repo in risk_maps/<repo>.json. Each matching rule sets MINIMUM
factor values, which are enforced after the LLM call (floors only ever raise).
A rule can carry stricter floors that apply only when the diff changes a
version in its files, or removes something from them.
"""

import json
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io_safety import read_text_file
from .preprocess import (
    ext_from_filename,
    filter_file,
    is_test_path,
    parse_diff_sections,
    redact,
    strip_comment_changes,
)

RISK_FACTORS = ("blast_radius", "severity", "hazard")
FACTOR_MAX = 5
RAW_MAX = FACTOR_MAX ** len(RISK_FACTORS)

_MAPS_DIR = Path(__file__).parent / "risk_maps"

# A changed line that sets a version-like key: Helm `version:` /
# `targetRevision:` / `tag:`, Terraform `version = "..."`, module `?ref=`,
# container `image: repo:tag`. Only consulted for rules that ask for it, so
# the pattern can stay loose.
_VERSION_LINE_RE = re.compile(
    r"(?i)(?:\b(?:version|targetrevision|revision|tag|appversion|chart|image)\b\s*[:=])"
    r"|(?:\?ref=)"
    r"|(?:\bv?\d+\.\d+\.\d+\b)"
)
# Terraform/HCL block removal: `-resource "x" "y" {`, `-module "x" {`.
_TF_BLOCK_REMOVAL_RE = re.compile(r'^-\s*(?:resource|module|data)\s+"')
# Deleting a lockfile destroys nothing it managed (infra#308 dropped 100
# .terraform.lock.hcl files), so it is not a removal.
_LOCKFILE_RE = re.compile(r"(?i)(?:^|/)[^/]*lock[^/]*$")


def load_risk_map(repo: str) -> List[Dict[str, Any]]:
    """Rules for a repo (bare name, e.g. `gitops`); empty if none is shipped."""
    path = _MAPS_DIR / f"{repo}.json"
    if not path.exists():
        return []
    return json.loads(read_text_file(path)).get("rules", [])


def file_signals(lines: List[str], removal_regex: Optional[str] = None) -> Dict[str, bool]:
    """
    Change signals for one file's diff lines (header + hunks).

    `removal` is a deleted file, a removed Terraform block, or — for files
    where removal means deleting a list entry (an addon from apps.yaml) — a
    removed line matching the rule's own `removal_regex`.
    """
    removed = [ln for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    version_change = any(_VERSION_LINE_RE.search(ln[1:]) for ln in removed) and any(
        _VERSION_LINE_RE.search(ln[1:]) for ln in added
    )
    deleted_file = any(ln.startswith("+++ /dev/null") for ln in lines) and not any(
        ln.startswith("--- a/") and _LOCKFILE_RE.search(ln) for ln in lines
    )
    removal = deleted_file or any(_TF_BLOCK_REMOVAL_RE.match(ln) for ln in removed)
    if removal_regex and not removal:
        entry_re = re.compile(removal_regex)
        removal = any(entry_re.search(ln[1:]) for ln in removed)
    return {"version_change": version_change, "removal": removal}


def match_rules(diff_text: str, rules: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    """
    Apply a risk map to a raw diff.

    Returns the per-factor floors (max across every matching rule) and one
    human-readable line per matched rule, for the prompt and the explanation.
    """
    floors = {k: 1 for k in RISK_FACTORS}
    reasons: List[str] = []
    if not rules:
        return floors, reasons

    # Same exclusions as the scored diff, plus every lockfile (filter_file
    # misses .terraform.lock.hcl): docs, lockfiles, tests and comment/blank-only
    # edits cannot put a floor under a PR.
    files = {}
    for fn, lines in parse_diff_sections(redact(diff_text), hunks_per_file=10**9).items():
        if not filter_file(fn) or is_test_path(fn) or _LOCKFILE_RE.search(fn):
            continue
        lines, adds, dels = strip_comment_changes(lines, ext_from_filename(fn))
        if adds + dels:
            files[fn] = lines

    for rule in rules:
        hits = [fn for fn in files if any(fnmatch(fn, pat) for pat in rule.get("paths", []))]
        if not hits:
            continue
        applied = {k: rule[k] for k in RISK_FACTORS if k in rule}
        tags = []
        signals = [file_signals(files[fn], rule.get("removal_regex")) for fn in hits]
        for signal, key in (("version_change", "on_version_change"), ("removal", "on_removal")):
            if key in rule and any(s[signal] for s in signals):
                tags.append(signal.replace("_", " "))
                for k, v in rule[key].items():
                    applied[k] = max(applied.get(k, 1), v)
        for k, v in applied.items():
            floors[k] = max(floors[k], int(v))
        mins = ", ".join(f"{k}>={v}" for k, v in applied.items())
        more = f" +{len(hits) - 1} more" if len(hits) > 1 else ""
        tag = f" [{', '.join(tags)}]" if tags else ""
        reasons.append(f"{hits[0]}{more}: {rule.get('reason', 'risk map rule')}{tag} ({mins})")
    return floors, reasons


def load_risk_prompt() -> str:
    return read_text_file(Path(__file__).parent / "prompt" / "risk.txt")


def build_risk_input(diff_excerpt: str, reasons: List[str]) -> str:
    """Prepend the risk-map context to the same excerpt the complexity call saw."""
    if not reasons:
        return diff_excerpt
    context = "\n".join(f"- {r}" for r in reasons)
    return f"CODEBASE RISK CONTEXT (from the repository's risk map):\n{context}\n\n{diff_excerpt}"


def parse_risk_response(response_text: str) -> Dict[str, Any]:
    """Parse the risk call's JSON; raises InvalidResponseError on bad shape."""
    from .scoring import InvalidResponseError

    text = response_text.strip()
    m = re.search(r"\{[^{}]*\"blast_radius\"[^{}]*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidResponseError(f"Failed to parse JSON: {e}")
    if not isinstance(data, dict):
        raise InvalidResponseError("Response is not a JSON object")
    factors = {}
    for k in RISK_FACTORS:
        if k not in data:
            raise InvalidResponseError(f"Missing risk factor: {k}")
        try:
            factors[k] = max(1, min(FACTOR_MAX, int(data[k])))
        except (ValueError, TypeError):
            raise InvalidResponseError(f"Invalid {k} value: {data[k]}")
    explanation = re.sub(r"\s+", " ", str(data.get("explanation", "")).strip())
    return {"factors": factors, "explanation": explanation}


def compose_risk(
    llm_factors: Dict[str, int],
    floors: Dict[str, int],
    max_score: int,
) -> Tuple[int, Dict[str, int]]:
    """Enforce floors, multiply, and rescale [1, RAW_MAX] onto [1, max_score]."""
    factors = {k: max(llm_factors.get(k, 1), floors.get(k, 1)) for k in RISK_FACTORS}
    raw = 1
    for v in factors.values():
        raw *= v
    score = round((raw - 1) * (max_score - 1) / (RAW_MAX - 1)) + 1
    return max(1, min(max_score, score)), factors


def format_risk_explanation(
    factors: Dict[str, int], llm_explanation: str, reasons: List[str]
) -> str:
    """One line: final factors, the model's rationale, then the rules that fired."""
    head = ", ".join(f"{k} {v}" for k, v in factors.items())
    out = f"[{head}] {llm_explanation}".strip()
    if reasons:
        out += " | risk map: " + "; ".join(reasons)
    return out


def assess_risk(
    provider: Any,
    repo: str,
    diff_text: str,
    diff_excerpt: str,
    stats_json: str,
    title: str,
    max_score: int,
    rules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the risk call and combine it with the repo's risk map."""
    if rules is None:
        rules = load_risk_map(repo)
    floors, reasons = match_rules(diff_text, rules)
    result = provider.assess_risk(
        prompt=load_risk_prompt(),
        diff_excerpt=build_risk_input(diff_excerpt, reasons),
        stats_json=stats_json,
        title=title,
    )
    score, factors = compose_risk(result["factors"], floors, max_score)
    return {
        "risk": score,
        "risk_factors": factors,
        "risk_llm_factors": result["factors"],
        "risk_explanation": format_risk_explanation(factors, result["explanation"], reasons),
        "risk_tokens": result.get("tokens"),
    }
