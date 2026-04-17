"""Tests for SkillsLoader — metadata parsing and baseDir substitution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_loader(workspace: Path, *, no_builtins: bool = True):
    from nanobot.agent.skills import SkillsLoader

    builtin_dir = workspace / "_no_builtins_"  # non-existent dir disables builtins
    return SkillsLoader(workspace, builtin_skills_dir=builtin_dir if no_builtins else None)


def _write_skill(workspace: Path, name: str, skill_md: str) -> Path:
    """Write a skill into workspace/skills/<name>/SKILL.md and return the skill dir."""
    skill_dir = workspace / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------------------
# _extract_multiline_json
# ---------------------------------------------------------------------------

def test_extract_multiline_json_parses_openclaw_style():
    from nanobot.agent.skills import SkillsLoader

    frontmatter = (
        'name: x-search\n'
        'description: Search X\n'
        'metadata:\n'
        '  {\n'
        '    "openclaw":\n'
        '      {\n'
        '        "requires": { "bins": ["python3"], "env": ["XAI_API_KEY"] }\n'
        '      }\n'
        '  }\n'
    )
    raw = SkillsLoader._extract_multiline_json(frontmatter, "metadata")
    import json
    parsed = json.loads(raw)
    assert parsed["openclaw"]["requires"]["env"] == ["XAI_API_KEY"]
    assert parsed["openclaw"]["requires"]["bins"] == ["python3"]


def test_extract_multiline_json_returns_empty_for_missing_key():
    from nanobot.agent.skills import SkillsLoader

    frontmatter = "name: foo\ndescription: bar\n"
    assert SkillsLoader._extract_multiline_json(frontmatter, "metadata") == ""


def test_extract_multiline_json_strips_trailing_commas():
    from nanobot.agent.skills import SkillsLoader

    frontmatter = (
        'metadata:\n'
        '  {\n'
        '    "openclaw": {\n'
        '      "emoji": "X",\n'
        '      "requires": { "env": ["KEY"], },\n'
        '    },\n'
        '  }\n'
    )
    raw = SkillsLoader._extract_multiline_json(frontmatter, "metadata")
    import json
    parsed = json.loads(raw)
    assert parsed["openclaw"]["requires"]["env"] == ["KEY"]


# ---------------------------------------------------------------------------
# get_skill_metadata with multi-line metadata
# ---------------------------------------------------------------------------

def test_get_skill_metadata_parses_multiline_metadata(tmp_path):
    skill_md = (
        '---\n'
        'name: x-search\n'
        'description: Search X posts\n'
        'metadata:\n'
        '  {\n'
        '    "openclaw":\n'
        '      {\n'
        '        "requires": { "bins": ["python3"], "env": ["XAI_API_KEY"] }\n'
        '      }\n'
        '  }\n'
        '---\n\n'
        '# X Search\n'
    )
    _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)
    meta = loader.get_skill_metadata("x-search")

    assert meta is not None
    assert meta["name"] == "x-search"
    assert meta["description"] == "Search X posts"
    assert meta["metadata"]  # non-empty JSON string


def test_get_skill_metadata_requirements_checked_from_multiline(tmp_path, monkeypatch):
    skill_md = (
        '---\n'
        'name: x-search\n'
        'description: Search X\n'
        'metadata:\n'
        '  {\n'
        '    "openclaw": {\n'
        '      "requires": { "bins": ["python3"], "env": ["XAI_API_KEY"] }\n'
        '    }\n'
        '  }\n'
        '---\n\n# X Search\n'
    )
    _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr("nanobot.agent.skills.shutil.which", lambda b: None)

    skill_meta = loader._get_skill_meta("x-search")
    assert not loader._check_requirements(skill_meta)

    skills = loader.list_skills(filter_unavailable=True)
    assert not any(s["name"] == "x-search" for s in skills)


def test_get_skill_metadata_available_when_requirements_met(tmp_path, monkeypatch):
    skill_md = (
        '---\n'
        'name: x-search\n'
        'description: Search X\n'
        'metadata:\n'
        '  {\n'
        '    "openclaw": {\n'
        '      "requires": { "bins": ["python3"], "env": ["XAI_API_KEY"] }\n'
        '    }\n'
        '  }\n'
        '---\n\n# X Search\n'
    )
    _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("nanobot.agent.skills.shutil.which", lambda b: f"/usr/bin/{b}")

    skill_meta = loader._get_skill_meta("x-search")
    assert loader._check_requirements(skill_meta)

    skills = loader.list_skills(filter_unavailable=True)
    assert any(s["name"] == "x-search" for s in skills)


# ---------------------------------------------------------------------------
# {baseDir} substitution
# ---------------------------------------------------------------------------

def test_load_skill_substitutes_base_dir(tmp_path):
    skill_md = (
        '---\nname: x-search\ndescription: test\n---\n\n'
        '```bash\npython3 {baseDir}/scripts/search.py "query"\n```\n'
    )
    skill_dir = _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)

    content = loader.load_skill("x-search")
    assert content is not None
    assert "{baseDir}" not in content
    assert skill_dir.as_posix() in content


def test_build_skills_summary_includes_base_dir(tmp_path):
    skill_md = '---\nname: x-search\ndescription: Search X\n---\n\n# X Search\n'
    skill_dir = _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)

    summary = loader.build_skills_summary()
    assert "<baseDir>" in summary
    assert skill_dir.as_posix() in summary


def test_build_skills_summary_uses_posix_paths(tmp_path):
    """Both <location> and <baseDir> must use forward slashes so the model
    can construct valid exec commands on all platforms."""
    skill_md = '---\nname: x-search\ndescription: Search X\n---\n\n# X Search\n'
    _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)

    summary = loader.build_skills_summary()
    # Extract the <location> and <baseDir> values and verify no backslashes
    import re
    location = re.search(r"<location>(.*?)</location>", summary)
    base_dir = re.search(r"<baseDir>(.*?)</baseDir>", summary)
    assert location and "\\" not in location.group(1)
    assert base_dir and "\\" not in base_dir.group(1)


def test_build_skills_summary_xml_escapes_location(tmp_path):
    """<location> path must be XML-escaped (consistent with <baseDir>)."""
    skill_md = '---\nname: x-search\ndescription: Search X\n---\n\n# X Search\n'
    _write_skill(tmp_path, "x-search", skill_md)
    loader = _make_loader(tmp_path)

    summary = loader.build_skills_summary()
    # Valid XML — should parse without error
    import xml.etree.ElementTree as ET
    ET.fromstring(summary)  # raises if invalid XML


# -- multiline description tests (YAML folded > and literal |) -----------------


def test_build_skills_summary_folded_description(tmp_path: Path) -> None:
    """description: > (YAML folded scalar) should be parsed correctly."""
    from nanobot.agent.skills import SkillsLoader

    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    skill_dir = ws_skills / "pdf"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: pdf\n"
        "description: >\n"
        "  Use this skill when visual quality and design identity matter for a PDF.\n"
        "  CREATE (generate from scratch): \"make a PDF\".\n"
        "---\n\n# PDF Skill\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    summary = loader.build_skills_summary()
    assert "pdf" in summary
    assert "visual quality" in summary


def test_build_skills_summary_literal_description(tmp_path: Path) -> None:
    """description: | (YAML literal scalar) should be parsed correctly."""
    from nanobot.agent.skills import SkillsLoader

    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    skill_dir = ws_skills / "multi"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: multi\n"
        "description: |\n"
        "  Line one of description.\n"
        "  Line two of description.\n"
        "---\n\n# Multi\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    meta = loader.get_skill_metadata("multi")
    assert meta is not None
    desc = meta.get("description")
    assert isinstance(desc, str)
    assert "Line one" in desc
    assert "Line two" in desc


def test_get_skill_metadata_handles_yaml_types(tmp_path: Path) -> None:
    """yaml.safe_load returns native types; always should be True, not 'true'."""
    from nanobot.agent.skills import SkillsLoader

    workspace = tmp_path / "ws"
    ws_skills = workspace / "skills"
    ws_skills.mkdir(parents=True)
    skill_dir = ws_skills / "typed"
    skill_dir.mkdir(parents=True)
    payload = json.dumps({"nanobot": {"requires": {"bins": ["gh"]}, "always": True}}, separators=(",", ":"))
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: typed\n"
        f"metadata: {payload}\n"
        "always: true\n"
        "---\n\n# Typed\n",
        encoding="utf-8",
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    meta = loader.get_skill_metadata("typed")
    assert meta is not None
    # YAML parsed 'true' to Python True
    assert meta.get("always") is True
    # metadata is a parsed dict, not a JSON string
    assert isinstance(meta.get("metadata"), dict)
