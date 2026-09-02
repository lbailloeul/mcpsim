"""Shared helpers for the heavy generation backends.

Toolchain capability checks, subprocess running, and on-demand C++ compilation
(mirroring run_lanl_12bar_pipeline.sh). Backends raise ToolchainError with an
actionable message when a required tool is missing, so the CLI can report it
cleanly instead of crashing.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class ToolchainError(RuntimeError):
    """A required external tool or Python module is unavailable."""


def have_command(name: str) -> bool:
    return shutil.which(name) is not None


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require_command(name: str, hint: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolchainError(f"'{name}' not found on PATH. {hint}")
    return path


def require_module(name: str, hint: str) -> None:
    if not have_module(name):
        raise ToolchainError(f"Python module '{name}' not importable. {hint}")


def run(cmd: Sequence[str], cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a command, streaming output; raise on non-zero exit."""
    printable = " ".join(str(c) for c in cmd)
    print(f"[mcpsim] $ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env)
    if proc.returncode != 0:
        raise ToolchainError(f"command failed (exit {proc.returncode}): {printable}")


def root_flags() -> list[str]:
    """`root-config --cflags --libs`, split into tokens."""
    require_command("root-config", "Source ROOT (thisroot.sh) before regenerating.")
    out = subprocess.check_output(["root-config", "--cflags", "--libs"], text=True)
    return out.split()


def pythia8_flags() -> list[str]:
    """`pythia8-config --cxxflags --libs`, split into tokens.

    Required to compile the PYTHIA meson generator. Raises if PYTHIA's config
    helper is absent so the caller can fall back / report cleanly.
    """
    require_command("pythia8-config",
                    "Install PYTHIA 8 (provides pythia8-config) before regenerating mesons.")
    out = subprocess.check_output(["pythia8-config", "--cxxflags", "--libs"], text=True)
    return out.split()


def compile_cpp(source: Path, out_binary: Path, extra_flags: Sequence[str] = (),
                with_pythia: bool = False) -> Path:
    """Compile a ROOT (+ optional PYTHIA) C++ source (c++ -std=c++17 ...)."""
    out_binary.parent.mkdir(parents=True, exist_ok=True)
    cxx = shutil.which("c++") or require_command("g++", "Install a C++17 compiler.")
    flags = [*root_flags()]
    if with_pythia:
        flags += pythia8_flags()
    cmd = [cxx, "-std=c++17", str(source), *flags, *extra_flags, "-o", str(out_binary)]
    run(cmd)
    return out_binary
