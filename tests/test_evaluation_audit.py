from pathlib import Path

import pytest

from src.evaluation.evaluation_audit import (
    audit_markdown_relative_links,
    audit_required_gitignore_patterns,
    build_evaluation_artifact_inventory,
    collect_markdown_files,
    extract_markdown_links,
    read_gitignore_patterns,
    summarize_directory_artifacts,
)


def test_summarize_directory_artifacts_missing_directory(tmp_path) -> None:
    missing_path = tmp_path / "missing"

    summary = summarize_directory_artifacts(missing_path)

    assert summary == {
        "directory": str(missing_path),
        "exists": False,
        "n_files": 0,
        "total_bytes": 0,
        "files": [],
    }


def test_summarize_directory_artifacts_counts_files_and_bytes(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    nested_dir = artifact_dir / "nested"
    nested_dir.mkdir(parents=True)
    (artifact_dir / "a.txt").write_bytes(b"abc")
    (nested_dir / "b.bin").write_bytes(b"12345")

    summary = summarize_directory_artifacts(artifact_dir)

    assert summary["exists"] is True
    assert summary["n_files"] == 2
    assert summary["total_bytes"] == 8
    assert summary["files"] == ["a.txt", "nested/b.bin"]


def test_summarize_directory_artifacts_filters_patterns_recursively(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    nested_dir = artifact_dir / "nested"
    nested_dir.mkdir(parents=True)
    (artifact_dir / "plot.png").write_bytes(b"png")
    (artifact_dir / "record.jsonl").write_bytes(b"jsonl")
    (nested_dir / "nested_plot.png").write_bytes(b"nested")

    summary = summarize_directory_artifacts(artifact_dir, patterns=["*.png"])

    assert summary["n_files"] == 2
    assert summary["total_bytes"] == 9
    assert summary["files"] == ["nested/nested_plot.png", "plot.png"]


def test_build_evaluation_artifact_inventory_summarizes_known_locations(
    tmp_path,
) -> None:
    figures_dir = tmp_path / "results" / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "figure.png").write_bytes(b"png")
    (figures_dir / "notes.txt").write_bytes(b"text")

    inventory = build_evaluation_artifact_inventory(tmp_path)

    assert sorted(inventory) == [
        "canvas_debug",
        "checkpoints",
        "figures",
        "registry",
        "user_digits",
    ]
    assert inventory["figures"]["files"] == ["figure.png"]
    assert inventory["figures"]["n_files"] == 1
    assert inventory["registry"]["exists"] is False


def test_read_gitignore_patterns_ignores_blank_lines_and_comments(tmp_path) -> None:
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text(
        "\n# comment\nresults/figures/*.png\n\n  data/user_digits/  \n",
        encoding="utf-8",
    )

    assert read_gitignore_patterns(gitignore_path) == [
        "results/figures/*.png",
        "data/user_digits/",
    ]


def test_audit_required_gitignore_patterns_detects_present_and_missing(
    tmp_path,
) -> None:
    gitignore_path = tmp_path / ".gitignore"
    gitignore_path.write_text(
        "results/figures/*.png\nresults/registry/\n",
        encoding="utf-8",
    )

    audit = audit_required_gitignore_patterns(
        gitignore_path,
        [
            "results/figures/*.png",
            "results/registry/",
            "data/user_digits/",
        ],
    )

    assert audit == {
        "gitignore_path": str(gitignore_path),
        "present": ["results/figures/*.png", "results/registry/"],
        "missing": ["data/user_digits/"],
        "all_present": False,
    }


def test_audit_required_gitignore_patterns_missing_file_reports_all_missing(
    tmp_path,
) -> None:
    gitignore_path = tmp_path / ".gitignore"
    required = ["results/figures/*.png", "data/user_digits/"]

    audit = audit_required_gitignore_patterns(gitignore_path, required)

    assert audit["present"] == []
    assert audit["missing"] == required
    assert audit["all_present"] is False


def test_extract_markdown_links_extracts_normal_links() -> None:
    assert extract_markdown_links("See [report](reports/week5.md).") == [
        "reports/week5.md"
    ]


def test_extract_markdown_links_ignores_image_links() -> None:
    assert extract_markdown_links("![diagram](figures/plot.png)") == []


def test_extract_markdown_links_handles_multiple_links() -> None:
    links = extract_markdown_links(
        "[one](one.md), ![image](image.png), and [two](two.md)"
    )

    assert links == ["one.md", "two.md"]


def test_extract_markdown_links_returns_empty_list_without_links() -> None:
    assert extract_markdown_links("Plain text with no Markdown links.") == []


def test_audit_markdown_relative_links_reports_only_missing_md_links(
    tmp_path,
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "valid.md").write_text("# Valid\n", encoding="utf-8")
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "[valid](reports/valid.md)",
                "[valid anchor](reports/valid.md#section)",
                "[missing](reports/missing.md)",
                "[external](https://example.com/missing.md)",
                "[mail](mailto:test@example.com)",
                "[page anchor](#local-section)",
                "[image target](reports/plot.png)",
            ]
        ),
        encoding="utf-8",
    )

    issues = audit_markdown_relative_links(tmp_path, [readme_path])

    assert issues == [
        {
            "file": str(readme_path),
            "target": "reports/missing.md",
            "resolved": str((reports_dir / "missing.md").resolve()),
            "issue": "missing",
        }
    ]


def test_audit_markdown_relative_links_handles_query_strings(tmp_path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "target.md").write_text("# Target\n", encoding="utf-8")
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "[target](docs/target.md?plain=1#heading)",
        encoding="utf-8",
    )

    assert audit_markdown_relative_links(tmp_path, [readme_path]) == []


def test_collect_markdown_files_includes_readme_and_reports_sorted(
    tmp_path,
) -> None:
    (tmp_path / "README.md").write_text("# README\n", encoding="utf-8")
    reports_dir = tmp_path / "reports"
    nested_dir = reports_dir / "nested"
    nested_dir.mkdir(parents=True)
    (reports_dir / "z.md").write_text("# Z\n", encoding="utf-8")
    (nested_dir / "a.md").write_text("# A\n", encoding="utf-8")
    (reports_dir / "notes.txt").write_text("notes\n", encoding="utf-8")

    markdown_files = collect_markdown_files(tmp_path)
    relative_paths = [
        path.relative_to(tmp_path).as_posix() for path in markdown_files
    ]

    assert relative_paths == [
        "README.md",
        "reports/nested/a.md",
        "reports/z.md",
    ]


def test_collect_markdown_files_can_exclude_readme(tmp_path) -> None:
    (tmp_path / "README.md").write_text("# README\n", encoding="utf-8")
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "report.md").write_text("# Report\n", encoding="utf-8")

    markdown_files = collect_markdown_files(tmp_path, include_readme=False)

    assert [path.relative_to(tmp_path).as_posix() for path in markdown_files] == [
        "reports/report.md"
    ]


def test_summarize_directory_artifacts_rejects_file_path(tmp_path) -> None:
    artifact_file = tmp_path / "artifact.png"
    artifact_file.write_bytes(b"png")

    with pytest.raises(NotADirectoryError):
        summarize_directory_artifacts(artifact_file)
