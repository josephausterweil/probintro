#!/usr/bin/env python3
"""Validate data/course.toml (the textbook -> course lecture-link map).

Hard errors (exit 1):
  * a [chapters] key that doesn't resolve to content/<key>.md or
    content/<key>/_index.md
  * a chapter mapped to a week with no [weeks] title

Advisory warnings (exit 0) when the course repo is reachable (the textbook
working copy normally lives nested inside it):
  * a chapter referenced in a week's PLAN.md "Textbook Chapters" section that
    is missing from [chapters], or mapped to a *different* week (often fine —
    review readings — but worth eyeballing)
  * pdf_weeks out of sync with the PDFs actually published in docs/slides/

Usage: python3 scripts/check_course_links.py [--course-repo PATH]
"""

import argparse
import csv
import re
import sys
import tomllib
from pathlib import Path

TEXTBOOK_ROOT = Path(__file__).resolve().parent.parent


def load_url_map():
    """old '/intro2/05_mixture_models/' -> new '/learning/mixture-models/'."""
    path = TEXTBOOK_ROOT / "scripts" / "url_map.csv"
    if not path.is_file():
        return {}
    with path.open() as f:
        return {row["old_path"]: row["new_path"] for row in csv.DictReader(f)}


def plan_chapter_refs(plan_text: str, url_map: dict) -> set:
    """Chapter keys (part/slug) referenced in a PLAN's Textbook Chapters section."""
    m = re.search(
        r"^##\s+Textbook Chapters\s*\n(.*?)(?=^##\s|\Z)", plan_text, re.S | re.M
    )
    if not m:
        return set()
    refs = set()
    for ref in re.findall(r"`([\w/\-.]+/[\w\-.]+?)(?:\.md)?/?`", m.group(1)):
        ref = ref.strip("/")
        if "/" not in ref:
            continue
        old = f"/{ref}/"
        if old in url_map:  # legacy intro/intro2/NN_ path
            refs.add(url_map[old].strip("/"))
        elif (TEXTBOOK_ROOT / "content" / ref.split("/")[0]).is_dir() and not re.match(
            r"(intro2?|t\d)/", ref
        ):  # part/slug form under a real content Part
            refs.add(ref)
    return refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--course-repo",
        type=Path,
        default=TEXTBOOK_ROOT.parent,
        help="course repo root (default: textbook's parent dir)",
    )
    args = ap.parse_args()

    data = tomllib.loads((TEXTBOOK_ROOT / "data" / "course.toml").read_text())
    chapters, weeks = data["chapters"], data["weeks"]
    pdf_weeks = set(data.get("pdf_weeks", []))
    errors, warnings = [], []

    for key, week in chapters.items():
        if not (
            (TEXTBOOK_ROOT / "content" / f"{key}.md").is_file()
            or (TEXTBOOK_ROOT / "content" / key / "_index.md").is_file()
        ):
            errors.append(f"[chapters] {key!r}: no content/{key}.md or _index.md")
        if str(week) not in weeks:
            errors.append(f"[chapters] {key!r}: week {week} has no [weeks] title")

    course = args.course_repo / "course"
    if course.is_dir():
        url_map = load_url_map()
        for plan in sorted(course.glob("week*/PLAN.md")):
            wn = int(re.match(r"week(\d{2})_", plan.parent.name).group(1))
            for key in plan_chapter_refs(plan.read_text(), url_map):
                mapped = chapters.get(key)
                if mapped is None:
                    warnings.append(
                        f"week {wn} PLAN references {key!r} — not in [chapters] "
                        "(fine for reference pages; add it if it's a companion chapter)"
                    )
                elif mapped != wn:
                    warnings.append(
                        f"week {wn} PLAN references {key!r}, mapped to week {mapped} "
                        "(fine if the PLAN entry is review reading)"
                    )
    else:
        warnings.append(f"course repo not found at {args.course_repo} — PLAN checks skipped")

    slides = args.course_repo / "docs" / "slides"
    if slides.is_dir():
        published = {int(m.group(1)) for p in slides.glob("week*.pdf")
                     if (m := re.match(r"week(\d{2})\.pdf", p.name))}
        for w in sorted(pdf_weeks - published):
            errors.append(f"pdf_weeks lists week {w} but docs/slides/week{w:02d}.pdf doesn't exist")
        for w in sorted(published - pdf_weeks):
            warnings.append(f"docs/slides/week{w:02d}.pdf exists but week {w} isn't in pdf_weeks")

    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"warn:  {w}")
    print(f"\n{len(chapters)} chapters mapped, {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
