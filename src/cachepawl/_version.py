"""Package version helpers."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

PACKAGE_NAME = "cachepawl"
LOCAL_VERSION = "0.0.0+local"


def get_version() -> str:
    """Return the installed package version, falling back to local project metadata."""

    try:
        return installed_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return _local_pyproject_version()


def _local_pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return LOCAL_VERSION

    text = pyproject.read_text()
    project_match = re.search(r"(?ms)^\[project\]\n(?P<body>.*?)(?:^\[|\Z)", text)
    if project_match is None:
        return LOCAL_VERSION

    version_match = re.search(
        r'^version = "(?P<version>[^"]+)"$',
        project_match.group("body"),
        flags=re.MULTILINE,
    )
    return LOCAL_VERSION if version_match is None else version_match.group("version")
