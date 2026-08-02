"""Project evaluation artifact and report-link audit utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_MARKDOWN_LINK_PATTERN = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\n]*)\)")
_EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:", "#")


def summarize_directory_artifacts(
    directory: str | Path,
    patterns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Summarize generated artifact files in a directory.
    """
    path = Path(directory)
    if not path.exists():
        return {
            "directory": str(path),
            "exists": False,
            "n_files": 0,
            "total_bytes": 0,
            "files": [],
        }

    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory.")

    if patterns is None:
        candidate_paths = path.rglob("*")
    else:
        matched_paths: set[Path] = set()
        for pattern in patterns:
            matched_paths.update(path.rglob(pattern))
        candidate_paths = iter(matched_paths)

    files = sorted(
        candidate for candidate in candidate_paths if candidate.is_file()
    )
    relative_files = [
        file_path.relative_to(path).as_posix() for file_path in files
    ]
    total_bytes = sum(file_path.stat().st_size for file_path in files)

    return {
        "directory": str(path),
        "exists": True,
        "n_files": len(files),
        "total_bytes": total_bytes,
        "files": relative_files,
    }


def build_evaluation_artifact_inventory(
    project_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Build an inventory for known local evaluation artifact locations.
    """
    root = Path(project_root)
    locations = {
        "figures": (root / "results" / "figures", ["*.png"]),
        "registry": (root / "results" / "registry", ["*.jsonl"]),
        "checkpoints": (root / "results" / "checkpoints", ["*.npz"]),
        "canvas_debug": (root / "results" / "canvas_debug", ["*.png"]),
        "user_digits": (root / "data" / "user_digits", ["*.npz"]),
    }

    return {
        name: summarize_directory_artifacts(directory, patterns)
        for name, (directory, patterns) in locations.items()
    }


def read_gitignore_patterns(gitignore_path: str | Path) -> list[str]:
    """
    Read non-empty, non-comment gitignore patterns.
    """
    path = Path(gitignore_path)
    if not path.exists():
        return []

    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)

    return patterns


def audit_required_gitignore_patterns(
    gitignore_path: str | Path,
    required_patterns: list[str],
) -> dict[str, Any]:
    """
    Check whether required ignore patterns are present.
    """
    path = Path(gitignore_path)
    patterns = read_gitignore_patterns(path)
    pattern_set = set(patterns)
    present = [
        pattern for pattern in required_patterns if pattern in pattern_set
    ]
    missing = [
        pattern for pattern in required_patterns if pattern not in pattern_set
    ]

    return {
        "gitignore_path": str(path),
        "present": present,
        "missing": missing,
        "all_present": len(missing) == 0,
    }


def extract_markdown_links(markdown_text: str) -> list[str]:
    """
    Extract non-image Markdown link targets from text.
    """
    return [
        match.group(1).strip()
        for match in _MARKDOWN_LINK_PATTERN.finditer(markdown_text)
    ]


def audit_markdown_relative_links(
    project_root: str | Path,
    markdown_files: list[str | Path],
) -> list[dict[str, str]]:
    """
    Check whether relative Markdown links point to existing files.
    """
    root = Path(project_root)
    issues = []

    for markdown_file in markdown_files:
        markdown_path = Path(markdown_file)
        if not markdown_path.is_absolute():
            markdown_path = root / markdown_path

        markdown_text = markdown_path.read_text(encoding="utf-8")
        for raw_target in extract_markdown_links(markdown_text):
            target = raw_target.strip()
            if not target or target.startswith(_EXTERNAL_LINK_PREFIXES):
                continue

            link_path = target.split("#", maxsplit=1)[0]
            link_path = link_path.split("?", maxsplit=1)[0]
            if not link_path or not link_path.endswith(".md"):
                continue

            resolved = (markdown_path.parent / link_path).resolve()
            if not resolved.exists():
                issues.append(
                    {
                        "file": str(markdown_path),
                        "target": target,
                        "resolved": str(resolved),
                        "issue": "missing",
                    }
                )

    return issues


def collect_markdown_files(
    project_root: str | Path,
    include_readme: bool = True,
) -> list[Path]:
    """
    Collect README and report Markdown files for link auditing.
    """
    root = Path(project_root)
    markdown_files = []

    readme_path = root / "README.md"
    if include_readme and readme_path.exists():
        markdown_files.append(readme_path)

    reports_path = root / "reports"
    if reports_path.exists():
        markdown_files.extend(
            path for path in reports_path.rglob("*.md") if path.is_file()
        )

    return sorted(markdown_files)
