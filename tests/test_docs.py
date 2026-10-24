"""The published docs' `mcpsim` commands and YAML configs match the CLI and the config schema."""

import re
import shlex
from pathlib import Path

import click
import pytest

from mcpsim import paths
from mcpsim.cli import cli
from mcpsim.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md"] + sorted(p for p in (ROOT / "docs").glob("*.md")
                                     if p.name != "handoff.md")     # local notes, not published
BLOCKS = [(doc.name, lang, body) for doc in DOCS
          for lang, body in re.findall(r"```(\w*)\n(.*?)```", doc.read_text(), re.S)]
COMMANDS = [(name, line) for name, lang, body in BLOCKS for raw in body.splitlines()
            if (line := raw.split("#", 1)[0].strip()).startswith("mcpsim ")]
CONFIGS = [(name, body) for name, lang, body in BLOCKS if lang in ("yaml", "yml")]


def test_the_docs_show_commands_and_a_config():
    assert len(COMMANDS) > 10 and CONFIGS


@pytest.mark.parametrize("doc,line", COMMANDS)
def test_documented_commands_parse(doc, line):
    words = shlex.split(line)[1:]
    cmd = cli
    while isinstance(cmd, click.Group) and words and not words[0].startswith("-"):
        cmd = cmd.get_command(click.Context(cmd), words.pop(0))
        assert cmd is not None, f"{doc}: no such command in `{line}`"
    try:
        cmd.make_context(cmd.name, words)
    except click.UsageError as exc:
        pytest.fail(f"{doc}: `{line}`: {exc.format_message()}")


@pytest.mark.parametrize("doc,body", CONFIGS, ids=[f"{d}#{i}" for i, (d, _) in enumerate(CONFIGS)])
def test_documented_configs_load(doc, body, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_source_override", paths._source_override)  # a source_repo key sets it
    path = tmp_path / "config.yaml"
    path.write_text(body)
    load_config(path)
