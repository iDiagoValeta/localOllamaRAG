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
        # Dependabot branches are generated, not authored: the shape it writes
        # is the exception, so its PRs reach develop without a rename.
        ("dependabot/npm_and_yarn/rag/web/frontend/vite-6.4.4", "develop"),
        ("dependabot/npm_and_yarn/rag/web/frontend/katex-0.18.2", "develop"),
        ("dependabot/pip/rag/uv-0.9.1", "develop"),
        ("dependabot/github_actions/actions/checkout-5", "develop"),
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
        # The dependabot exception stops at develop: a dependency bump is not a
        # release, and it only covers branch names the bot actually generates.
        ("dependabot/npm_and_yarn/rag/web/frontend/vite-6.4.4", "main"),
        ("dependabot/npm_and_yarn/rag/web/frontend/vite-6.4.4", "rag-lite"),
        ("dependabot/npm_and_yarn/rag/web/frontend/vite", "develop"),
        ("dependabot/mi_gestor/rag/web/frontend/vite-6.4.4", "develop"),
        ("dependabot/issue-12-a-mano", "develop"),
    ],
)
def test_rejected_pairs(head, base):
    assert check(head, base) is not None


def test_cli_exit_codes(capsys):
    assert main(["x", "feat/issue-1-a", "develop"]) == 0
    assert main(["x", "feat/issue-1-a", "main"]) == 1
    assert main(["x"]) == 2
    assert "docs/standards/branches.md" in capsys.readouterr().out
