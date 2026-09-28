"""Tests for the risk module (PLT-4017)."""

import json

import pytest

from cli.risk import (
    RAW_MAX,
    compose_risk,
    file_signals,
    format_risk_explanation,
    load_risk_map,
    match_rules,
    parse_risk_response,
)
from cli.scoring import InvalidResponseError


def diff(path, removed, added, deleted=False):
    """Minimal unified diff for one file."""
    plus = "/dev/null" if deleted else f"b/{path}"
    body = "".join(f"-{ln}\n" for ln in removed) + "".join(f"+{ln}\n" for ln in added)
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ {plus}\n"
        f"@@ -1,{len(removed)} +1,{len(added)} @@\n{body}"
    )


KEDA_PROD_BUMP = diff(
    "aws/prod/prod-us-1/infra/apps.yaml", ["    version: 2.19.0"], ["    version: 2.21.0"]
)
KEDA_DEV_BUMP = diff(
    "aws/dev/us-east-1/infra/apps.yaml", ["    version: 2.19.0"], ["    version: 2.21.0"]
)


class TestSignals:
    def test_version_change_needs_both_sides(self):
        lines = KEDA_PROD_BUMP.splitlines()
        assert file_signals(lines)["version_change"]
        added_only = diff("x.yaml", [], ["    version: 2.21.0"]).splitlines()
        assert not file_signals(added_only)["version_change"]

    def test_terraform_block_removal(self):
        lines = diff("main.tf", ['resource "aws_route53_record" "clerk" {'], []).splitlines()
        assert file_signals(lines)["removal"]

    def test_deleted_file_is_removal(self):
        lines = diff("a.yaml", ["x: 1"], [], deleted=True).splitlines()
        assert file_signals(lines)["removal"]

    def test_removal_regex_list_entry(self):
        lines = diff("apps.yaml", ["  - name: keda", "    chart: keda"], []).splitlines()
        assert not file_signals(lines)["removal"]
        assert file_signals(lines, r"^\s*-\s+name:")["removal"]


class TestShippedMaps:
    @pytest.mark.parametrize("repo", ["gitops", "infrastructure", "platform"])
    def test_maps_load_and_are_well_formed(self, repo):
        rules = load_risk_map(repo)
        assert rules
        for rule in rules:
            assert rule["paths"] and rule["reason"]
            for key in ("blast_radius", "severity", "hazard"):
                assert 1 <= rule.get(key, 1) <= 5
            for extra in ("on_version_change", "on_removal"):
                for v in rule.get(extra, {}).values():
                    assert 1 <= v <= 5

    def test_unknown_repo_has_no_rules(self):
        assert load_risk_map("no-such-repo") == []

    def test_prod_keda_bump_floors_high(self):
        floors, reasons = match_rules(KEDA_PROD_BUMP, load_risk_map("gitops"))
        assert floors == {"blast_radius": 4, "severity": 4, "hazard": 4}
        assert any("version change" in r for r in reasons)
        score, _ = compose_risk({"blast_radius": 1, "severity": 1, "hazard": 1}, floors, 100)
        # Above the auto-approve risk gate (40): a one-line prod controller
        # upgrade no longer auto-approves on low complexity.
        assert score > 40

    def test_dev_keda_bump_is_lower_than_prod(self):
        rules = load_risk_map("gitops")
        dev, _ = compose_risk({}, match_rules(KEDA_DEV_BUMP, rules)[0], 100)
        prod, _ = compose_risk({}, match_rules(KEDA_PROD_BUMP, rules)[0], 100)
        assert 1 < dev < prod

    def test_prod_tenant_deletion_is_destructive(self):
        d = diff("aws/prod/prod-us-1/sun/customers/forter.yaml", ["name: forter"], [], True)
        floors, _ = match_rules(d, load_risk_map("gitops"))
        assert floors["hazard"] == 5 and floors["severity"] == 5

    def test_infra_prod_rds_change(self):
        d = diff(
            "terragrunt/aws/prod/prod-us-1/rds-proxy/terragrunt.hcl",
            ["  max_connections_percent = 50"],
            ["  max_connections_percent = 90"],
        )
        floors, _ = match_rules(d, load_risk_map("infrastructure"))
        assert floors["blast_radius"] == 4 and floors["severity"] == 4

    def test_test_and_markdown_files_do_not_match(self):
        d = diff("aws/prod/prod-us-1/infra/README.md", ["a"], ["b"])
        floors, reasons = match_rules(d, load_risk_map("gitops"))
        assert reasons == [] and floors == {"blast_radius": 1, "severity": 1, "hazard": 1}


class TestCompose:
    def test_bounds(self):
        assert compose_risk({"blast_radius": 1, "severity": 1, "hazard": 1}, {}, 100)[0] == 1
        top = {"blast_radius": 5, "severity": 5, "hazard": 5}
        assert compose_risk(top, {}, 100)[0] == 100
        assert RAW_MAX == 125

    def test_floors_only_raise(self):
        score, factors = compose_risk(
            {"blast_radius": 5, "severity": 2, "hazard": 3}, {"blast_radius": 3, "hazard": 4}, 100
        )
        assert factors == {"blast_radius": 5, "severity": 2, "hazard": 4}
        assert score == round((5 * 2 * 4 - 1) * 99 / 124) + 1

    def test_explanation_lists_rules(self):
        text = format_risk_explanation(
            {"blast_radius": 4, "severity": 4, "hazard": 4}, "KEDA upgrade", ["apps.yaml: prod"]
        )
        assert text.startswith("[blast_radius 4, severity 4, hazard 4] KEDA upgrade")
        assert "risk map: apps.yaml: prod" in text


class TestParse:
    def test_valid_and_clamped(self):
        r = parse_risk_response(
            'noise {"blast_radius": 9, "severity": 0, "hazard": 3, "explanation": "a\\nb"}'
        )
        assert r["factors"] == {"blast_radius": 5, "severity": 1, "hazard": 3}
        assert r["explanation"] == "a b"

    def test_missing_factor(self):
        with pytest.raises(InvalidResponseError):
            parse_risk_response(json.dumps({"blast_radius": 2, "severity": 2}))


def test_analyze_pr_to_dict_uses_separate_risk_call(monkeypatch):
    """Risk comes from assess_risk + the map, not the complexity call's risk dimension."""
    import cli.main as main

    monkeypatch.setenv("COMPLEXITY_MAX_SCORE", "100")
    meta = {"title": "KEDA 2.21.0 on prod", "additions": 1, "deletions": 1, "changed_files": 1}
    monkeypatch.setattr(main, "fetch_pr", lambda *a, **k: (KEDA_PROD_BUMP, meta))

    class FakeProvider:
        def __init__(self, *a, **k):
            pass

        def analyze_complexity(self, **kwargs):
            dims = {"scope": 1, "logic": 1, "integration": 1, "testing": 1, "risk": 1}
            return {"complexity": 3, "explanation": "bump", "dimensions": dims, "risk": 1}

        def assess_risk(self, **kwargs):
            assert "CODEBASE RISK CONTEXT" in kwargs["diff_excerpt"]
            return {"factors": {"blast_radius": 4, "severity": 3, "hazard": 3}, "explanation": "x"}

    monkeypatch.setattr(main, "OpenAIProvider", FakeProvider)
    out = main.analyze_pr_to_dict(
        "https://github.com/sun-security/gitops/pull/679", "prompt", None, "key"
    )
    assert out["score"] == 3
    assert out["risk_factors"] == {"blast_radius": 4, "severity": 4, "hazard": 4}
    assert out["risk"] > 40
    assert "risk map:" in out["risk_explanation"]


def test_comment_only_change_sets_no_floor():
    d = diff("aws/prod/prod-us-1/infra/apps.yaml", ["# old note"], ["# new note"])
    floors, reasons = match_rules(d, load_risk_map("gitops"))
    assert reasons == [] and floors == {"blast_radius": 1, "severity": 1, "hazard": 1}


def test_deleted_lockfile_is_not_removal():
    lines = diff(
        "terragrunt/aws/prod/prod-us-1/eks/.terraform.lock.hcl", ['provider "x" {'], [], True
    ).splitlines()
    assert not file_signals(lines)["removal"]
