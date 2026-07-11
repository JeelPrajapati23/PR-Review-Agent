"""Score whatever's currently stored in evaluation/results/ against each
fixture's meta.json expectations. Purely local/offline -- no Groq or Redis
calls -- so it can be run at any time, including partway through
run_reviews.py's multi-day resume cycle, to check progress, and again once
all 15 results are in for the final read.

Scoring is a simple grounding heuristic, not a semantic judge: a fixture
counts as "caught" only if the synthesized review text mentions both the
file basename meta.json names in must_flag_file AND at least one of its
issue_keywords. That mirrors app/agent.py's own _is_grounded() philosophy
(a real finding names its file) rather than trusting free-form prose alone.

Usage:
    python evaluation/evaluate_results.py
"""
import argparse
import json
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DATASET_ROOT / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory (relative to evaluation/) to read results from and write the summary into",
    )
    return parser.parse_args()


def _score_entry(entry: dict, record: dict | None) -> dict:
    fixture_id = entry["id"]
    if record is None:
        return {
            "id": fixture_id,
            "category": entry["category"],
            "status": "not_reviewed",
            "caught": None,
            "matched_keywords": [],
        }

    agent_result = record.get("agent_result", {})
    status = agent_result.get("status", "unknown")
    expected = record.get("expected_findings", {})

    if status != "completed":
        return {
            "id": fixture_id,
            "category": entry["category"],
            "status": status,
            "caught": None,
            "matched_keywords": [],
            "error": agent_result.get("error"),
        }

    summary_text = (agent_result.get("summary") or "").lower()
    must_flag_file = expected.get("must_flag_file", "").lower()
    file_flagged = bool(must_flag_file) and must_flag_file in summary_text
    matched_keywords = [kw for kw in expected.get("issue_keywords", []) if kw.lower() in summary_text]

    return {
        "id": fixture_id,
        "category": entry["category"],
        "status": status,
        "file_flagged": file_flagged,
        "matched_keywords": matched_keywords,
        "caught": file_flagged and bool(matched_keywords),
        "severity": expected.get("severity"),
    }


def _category_stats(scored: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    for row in scored:
        cat = stats.setdefault(
            row["category"], {"total": 0, "reviewed": 0, "caught": 0, "not_reviewed": 0, "no_finding": 0}
        )
        cat["total"] += 1
        if row["status"] == "not_reviewed":
            cat["not_reviewed"] += 1
            continue
        cat["reviewed"] += 1
        if row["status"] != "completed":
            continue
        if row["caught"]:
            cat["caught"] += 1
        else:
            cat["no_finding"] += 1
    return stats


def _render_markdown(scored: list[dict], stats: dict) -> str:
    lines = ["# Golden dataset evaluation results\n"]
    total_caught = sum(s["caught"] for s in stats.values())
    total_reviewed = sum(s["reviewed"] for s in stats.values())
    total = sum(s["total"] for s in stats.values())
    lines.append(f"**Overall: {total_caught}/{total_reviewed} caught ({total_reviewed}/{total} reviewed so far)**\n")

    for category, cat_stats in stats.items():
        lines.append(
            f"## {category} -- {cat_stats['caught']}/{cat_stats['reviewed']} caught "
            f"({cat_stats['reviewed']}/{cat_stats['total']} reviewed)\n"
        )
        lines.append("| id | status | caught | matched keywords |")
        lines.append("|---|---|---|---|")
        for row in scored:
            if row["category"] != category:
                continue
            caught_display = "-" if row["caught"] is None else ("yes" if row["caught"] else "no")
            keywords_display = ", ".join(row.get("matched_keywords", [])) or "-"
            lines.append(f"| {row['id']} | {row['status']} | {caught_display} | {keywords_display} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    results_dir = DATASET_ROOT / args.results_dir
    summary_json_path = results_dir / "_summary.json"
    summary_md_path = results_dir / "_summary.md"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]

    scored = []
    for entry in entries:
        result_path = results_dir / f"{entry['id']}.json"
        record = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        scored.append(_score_entry(entry, record))

    stats = _category_stats(scored)

    results_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps({"per_fixture": scored, "per_category": stats}, indent=2), encoding="utf-8"
    )
    markdown = _render_markdown(scored, stats)
    summary_md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nWritten to {summary_json_path} and {summary_md_path}")


if __name__ == "__main__":
    main()
