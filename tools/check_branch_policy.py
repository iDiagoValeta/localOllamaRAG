"""Check that a pull request's head and base branches follow the gitflow policy.

The policy lives in ``docs/standards/branches.md``; this module is its
executable half, so that a PR from ``feat/...`` straight into ``main`` fails a
required check instead of depending on a reviewer noticing the base branch.

Standard library only: the fast gate's ``architecture`` job installs nothing.

Usage:
    python tools/check_branch_policy.py <head-branch> <base-branch>
"""

import re
import sys

_TOPIC = re.compile(r"^(feat|fix|cleanup|probe|docs)/issue-\d+-[a-z0-9][a-z0-9-]*$")
_RELEASE = re.compile(r"^release/v\d+\.\d+\.\d+$")
_HOTFIX = re.compile(r"^hotfix/v\d+\.\d+\.\d+$")

# rag-lite is a long-lived variant (AGENTS.md section 3), not part of the
# release line: it diverged from main and receives its own topic branches.
_VARIANT_BASES = ("rag-lite",)


def check(head: str, base: str) -> str | None:
    """Return why ``head`` may not merge into ``base``, or ``None`` if it may.

    Args:
        head: Source branch name of the pull request.
        base: Target branch name of the pull request.

    Returns:
        A one-line reason when the pair violates the policy, otherwise None.
    """
    if base == "main":
        if _RELEASE.match(head) or _HOTFIX.match(head):
            return None
        return (
            f"'{head}' cannot target main: main only receives release/vX.Y.Z or "
            "hotfix/vX.Y.Z. Open the PR against develop."
        )
    if base == "develop":
        if _TOPIC.match(head) or _RELEASE.match(head) or _HOTFIX.match(head):
            return None
        if head == "main":
            return None
        return (
            f"'{head}' is not a valid branch name for develop: use "
            "<feat|fix|cleanup|probe|docs>/issue-<n>-<slug> (issue-0 only when no "
            "issue exists), or release/hotfix for the back-merge."
        )
    if _RELEASE.match(base):
        if re.match(r"^(fix|docs)/issue-\d+-[a-z0-9][a-z0-9-]*$", head):
            return None
        return f"'{head}' cannot target {base}: a release branch only takes fix/ and docs/ branches."
    if base in _VARIANT_BASES:
        if _TOPIC.match(head) or head in ("develop", "main"):
            return None
        return f"'{head}' is not a valid branch name for {base}."
    return f"base '{base}' is not a long-lived branch of the gitflow policy."


def main(argv: list[str]) -> int:
    """Run the check from the command line; exit status 1 on a violation."""
    if len(argv) != 3:
        print(__doc__)
        return 2
    reason = check(argv[1], argv[2])
    if reason:
        print(f"Branch policy violation: {reason}")
        print("See docs/standards/branches.md.")
        return 1
    print(f"OK: {argv[1]} -> {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
