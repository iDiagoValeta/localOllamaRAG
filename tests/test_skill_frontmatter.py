"""Every versioned runbook's frontmatter parses as YAML, with the fields the agents read (issue #606).

Lives with the loose tests because it needs PyYAML, which the engine job installs
(through ``rag/requirements.txt``) and the stdlib-only architecture job does not. A
frontmatter that does not parse is not reported anywhere: Claude Code and Codex
just skip the skill. ``run-local`` was rejected from #295 (2026-10-16) until
#606 over one unquoted ``: `` in its description.
"""

from pathlib import Path

import pytest
import yaml

SKILLS = sorted((Path(__file__).resolve().parents[1] / ".claude" / "skills").glob("*/SKILL.md"))


def test_there_are_skills_to_check():
    assert SKILLS, "no .claude/skills/*/SKILL.md found: the glob, not the skills, is broken"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda path: path.parent.name)
def test_frontmatter_parses_and_names_the_skill(skill):
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "frontmatter must open on the first line"
    frontmatter = yaml.safe_load(text.split("---\n", 2)[1])
    assert frontmatter["name"] == skill.parent.name
    assert isinstance(frontmatter["description"], str) and frontmatter["description"].strip()
