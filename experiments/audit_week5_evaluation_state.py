"""Audit Week 5 evaluation artifacts, ignore rules, and report links."""

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluation_audit import (
    audit_markdown_relative_links,
    audit_required_gitignore_patterns,
    build_evaluation_artifact_inventory,
    collect_markdown_files,
)


REQUIRED_GITIGNORE_PATTERNS = [
    "results/figures/*.png",
    "results/checkpoints/*.npz",
    "results/canvas_debug/",
    "results/registry/",
    "data/user_digits/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local evaluation artifacts and report links.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root to audit (default: current directory).",
    )
    parser.add_argument(
        "--fail-on-link-issues",
        action="store_true",
        help="Exit with code 1 when missing Markdown links are found.",
    )
    parser.add_argument(
        "--fail-on-missing-ignore",
        action="store_true",
        help="Exit with code 1 when required .gitignore patterns are missing.",
    )
    return parser.parse_args()


def _print_artifact_inventory(
    inventory: dict[str, dict[str, Any]],
) -> None:
    print("Artifact inventory")
    print("------------------")
    for name, summary in inventory.items():
        print(f"{name}: {summary['directory']}")
        print(f"  exists: {summary['exists']}")
        print(f"  n_files: {summary['n_files']}")
        print(f"  total_bytes: {summary['total_bytes']}")
        print("")


def _print_gitignore_audit(audit: dict[str, Any]) -> None:
    print("Gitignore audit")
    print("---------------")
    print(f"path: {audit['gitignore_path']}")
    print(f"all_present: {audit['all_present']}")
    if audit["present"]:
        print("present:")
        for pattern in audit["present"]:
            print(f"  - {pattern}")
    if audit["missing"]:
        print("missing:")
        for pattern in audit["missing"]:
            print(f"  - {pattern}")
    print("")


def _print_markdown_link_audit(
    markdown_files: list[Path],
    issues: list[dict[str, str]],
) -> None:
    print("Markdown link audit")
    print("-------------------")
    print(f"markdown_files_checked: {len(markdown_files)}")
    print(f"missing_relative_md_links: {len(issues)}")
    if issues:
        for issue in issues:
            print(
                (
                    f"  - {issue['file']} -> {issue['target']} "
                    f"({issue['issue']}: {issue['resolved']})"
                )
            )
    else:
        print("no missing relative Markdown links found")
    print("")


def _print_summary(
    inventory: dict[str, dict[str, Any]],
    gitignore_audit: dict[str, Any],
    markdown_issues: list[dict[str, str]],
) -> None:
    total_artifact_files = sum(
        int(summary["n_files"]) for summary in inventory.values()
    )
    total_artifact_bytes = sum(
        int(summary["total_bytes"]) for summary in inventory.values()
    )

    print("Summary")
    print("-------")
    print(f"artifact_locations: {len(inventory)}")
    print(f"artifact_files: {total_artifact_files}")
    print(f"artifact_bytes: {total_artifact_bytes}")
    print(f"missing_gitignore_patterns: {len(gitignore_audit['missing'])}")
    print(f"missing_markdown_links: {len(markdown_issues)}")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    inventory = build_evaluation_artifact_inventory(root)
    gitignore_audit = audit_required_gitignore_patterns(
        root / ".gitignore",
        REQUIRED_GITIGNORE_PATTERNS,
    )
    markdown_files = collect_markdown_files(root)
    markdown_issues = audit_markdown_relative_links(root, markdown_files)

    print(f"Project root: {root}")
    print("")
    _print_artifact_inventory(inventory)
    _print_gitignore_audit(gitignore_audit)
    _print_markdown_link_audit(markdown_files, markdown_issues)
    _print_summary(inventory, gitignore_audit, markdown_issues)

    should_fail_for_links = args.fail_on_link_issues and len(markdown_issues) > 0
    should_fail_for_ignores = (
        args.fail_on_missing_ignore and len(gitignore_audit["missing"]) > 0
    )
    if should_fail_for_links or should_fail_for_ignores:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
