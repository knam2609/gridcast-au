"""Run provenance: git commit and working-tree state, recorded with every model run."""

from __future__ import annotations

import subprocess

from gridcast.config import PROJECT_ROOT


def git_sha() -> str | None:
    """``<sha>`` or ``<sha>-dirty``; ``None`` outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha
