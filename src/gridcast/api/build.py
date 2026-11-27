"""Which code the API runs: the commit, and whether its files differed from it.

In the image, ``make api-up`` passes both as build arguments, which become
``GRIDCAST_BUILD_GIT_SHA`` and ``GRIDCAST_BUILD_DIRTY`` (m5_plan.md §4, step 4; the dirty flag
counts untracked files among those the image copies). In a checkout (``make api``) they come
from git, as every model run's provenance does (tracked files only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from gridcast.config import Settings
from gridcast.provenance import git_sha

Source = Literal["image", "checkout", "unknown"]


@dataclass(frozen=True, slots=True)
class BuildInfo:
    git_sha: str | None
    dirty: bool | None
    source: Source


def build_info(settings: Settings, checkout: Callable[[], str | None] = git_sha) -> BuildInfo:
    """The image's build arguments if it has them, else the checkout's git state."""
    if settings.build_git_sha is not None:
        return BuildInfo(settings.build_git_sha, settings.build_dirty, "image")
    sha = checkout()
    if sha is None:
        return BuildInfo(None, None, "unknown")
    return BuildInfo(sha.removesuffix("-dirty"), sha.endswith("-dirty"), "checkout")
