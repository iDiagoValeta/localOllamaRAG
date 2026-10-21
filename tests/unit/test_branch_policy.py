"""Gitflow branch policy: which head branches may target which base.

The pairs are the contract of docs/standards/branches.md. Every rejected pair
below is a mistake that is easy to make by hand (a topic branch opened against
main, a release branch opened against develop with a free-form name).
"""

import pytest

from tools.check_branch_policy import check, main


@pytest.mark.parametrize(
    ("head", "base"),
    [
        ("feat/issue-12-nuevo-store", "develop"),
        ("fix/issue-0-typo", "develop"),
        ("docs/issue-479-gitflow", "develop"),
        ("release/v4.1.0", "main"),
        ("hotfix/v4.0.2", "main"),
        ("release/v4.1.0", "develop"),
        ("hotfix/v4.0.2", "develop"),
        ("main", "develop"),
        ("fix/issue-5-x", "release/v4.1.0"),
        ("feat/issue-9-lite", "rag-lite"),
        ("develop", "rag-lite"),
    ],
)
def test_allowed_pairs(head, base):
    assert check(head, base) is None


@pytest.mark.parametrize(
    ("head", "base"),
    [
        ("feat/issue-12-nuevo-store", "main"),
        ("develop", "main"),
        ("release/4.1.0", "main"),
        ("hotfix/v4.0", "main"),
        ("feature/issue-12-x", "develop"),
        ("feat/sin-numero", "develop"),
        ("feat/issue-12-Mayusculas", "develop"),
        ("feat/issue-12-x", "release/v4.1.0"),
        ("feat/issue-12-x", "otra-rama"),
    ],
)
def test_rejected_pairs(head, base):
    assert check(head, base) is not None


def test_cli_exit_codes(capsys):
    assert main(["x", "feat/issue-1-a", "develop"]) == 0
    assert main(["x", "feat/issue-1-a", "main"]) == 1
    assert main(["x"]) == 2
    assert "docs/standards/branches.md" in capsys.readouterr().out
