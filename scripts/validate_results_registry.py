"""Cross-check RESULTS.md against the actual results/ directory.

RESULTS.md is the human-readable release index; each row that claims a
completed run should link to a real, parseable JSON artifact under
``results/``. This catches two release-blocking problems: a documented link
that points at a missing or malformed file, and a completed result artifact
that exists on disk but was never added to the registry (so it wouldn't ship
in release notes). Smoke-run and evaluation-scaffolding artifacts (paths
containing ``smoke`` or ``_eval``) are not required to be documented, since
they are gates, not reported results.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"\]\((results/[^)\s]+)\)")
UNDOCUMENTED_EXEMPT = ("smoke", "_eval")


def linked_result_paths(results_md_text: str) -> list[str]:
    return LINK_PATTERN.findall(results_md_text)


def validate_links(paths: list[str], *, root: Path = ROOT) -> None:
    """Raise if any RESULTS.md link is duplicated, escapes results/, missing, or not parseable JSON."""
    results_root = (root / "results").resolve()
    seen: set[str] = set()
    for relative_path in paths:
        if relative_path in seen:
            raise ValueError(f"RESULTS.md links the same path more than once: {relative_path}")
        seen.add(relative_path)
        full_path = (root / relative_path).resolve()
        if full_path != results_root and results_root not in full_path.parents:
            raise ValueError(f"RESULTS.md link escapes results/: {relative_path}")
        if not full_path.is_file():
            raise ValueError(f"RESULTS.md links a missing file: {relative_path}")
        try:
            payload = json.loads(full_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"RESULTS.md links invalid JSON: {relative_path}") from error
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"RESULTS.md links an empty or non-object JSON document: {relative_path}")


def find_undocumented_results(*, root: Path = ROOT, linked_paths: list[str]) -> list[str]:
    """Return real result artifacts on disk that RESULTS.md never links to."""
    linked = {(root / path).resolve() for path in linked_paths}
    results_root = root / "results"
    if not results_root.is_dir():
        return []
    candidates = list(results_root.glob("*.json")) + list(results_root.glob("*/result.json"))
    undocumented = []
    for candidate in sorted(candidates):
        if candidate.resolve() in linked:
            continue
        relative = candidate.relative_to(root).as_posix()
        if any(marker in relative for marker in UNDOCUMENTED_EXEMPT):
            continue
        undocumented.append(relative)
    return undocumented


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-md", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()
    text = args.results_md.read_text(encoding="utf-8")
    paths = linked_result_paths(text)
    validate_links(paths, root=ROOT)
    undocumented = find_undocumented_results(root=ROOT, linked_paths=paths)
    print(f"validated {len(paths)} RESULTS.md links")
    if undocumented:
        print("undocumented result artifacts (add a RESULTS.md row or exclude deliberately):")
        for path in undocumented:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
