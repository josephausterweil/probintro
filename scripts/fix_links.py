#!/usr/bin/env python3
"""Rewrite internal markdown links after the content restructure.

For every content/**/*.md (EN and .ja.md):
  1. Work out the file's OLD and NEW site URLs (via scripts/url_map.csv;
     files that didn't move have old == new).
  2. For every inline markdown link/image `](target)`:
       - external / mailto / pure-anchor targets are left alone;
       - `.md`-style targets are converted to URL-style;
       - the target is resolved against the file's OLD URL to an absolute
         site path, mapped through url_map (plus a small legacy-fix table
         for links that were already dead before the restructure), and
         re-relativized against the file's NEW URL;
       - a Japanese page whose relative link resolves into /ja/images/,
         /ja/widgets/ or /ja/notebooks/ (a long-standing bug: static files
         are not language-prefixed) is repaired to point at the real,
         un-prefixed static path.

Idempotent: running twice is a no-op. Prints a per-file change count.
"""

import csv
import posixpath
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

CONTENT = Path("content")
URL_MAP_CSV = Path("scripts/url_map.csv")

# Links that were already broken BEFORE the restructure, mapped to their
# intended targets in the new tree (see check_links.py baseline).
LEGACY_FIXES = {
    "/intro2/04_bayesian_gaussian/": "/learning/bayesian-learning/",
    "/genjax/05_bayesian_learning/": "/genjax/inference-in-action/",
    "/04_bayesian_learning/": "/learning/bayesian-learning/",
    "/genjax/05_inference/": "/genjax/inference-in-action/",
    "/genjax/00b_local_install/": "/genjax/local-install/",
}

STATIC_PREFIXES = ("/images/", "/widgets/", "/notebooks/", "/css/", "/js/")

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def load_map():
    old2new = {}
    with URL_MAP_CSV.open() as f:
        for row in csv.DictReader(f):
            old2new[row["old_path"]] = row["new_path"]
    old2new.update(LEGACY_FIXES)
    return old2new


def url_of(md: Path) -> str:
    """Site URL (EN space, no lang prefix) of a content file at its location."""
    rel = md.relative_to(CONTENT).as_posix()
    rel = re.sub(r"\.ja\.md$", ".md", rel)
    if rel.endswith("_index.md"):
        rel = rel[: -len("_index.md")]
    elif rel.endswith(".md"):
        rel = rel[: -len(".md")] + "/"
    return "/" + rel.lstrip("/")


def md_target_to_url(target: str, anchor: str):
    """Convert a `foo.md` / `foo.md/` style link target to URL-style."""
    t = target.rstrip("/")
    if t.endswith("_index.md"):
        t = t[: -len("_index.md")]
    elif t.endswith(".md"):
        t = t[: -len(".md")] + "/"
    return t, anchor


def rewrite_file(md: Path, old2new: dict, new2old: dict) -> int:
    is_ja = md.name.endswith(".ja.md")
    new_url_en = url_of(md)
    old_url_en = new2old.get(new_url_en, new_url_en)
    lang = "/ja" if is_ja else ""
    old_base = lang + old_url_en
    new_base = lang + new_url_en

    text = md.read_text(encoding="utf-8")
    changes = 0

    def sub(m):
        nonlocal changes
        raw = m.group(1)
        parsed = urlparse(raw)
        if parsed.scheme or parsed.netloc:
            return m.group(0)
        if raw.startswith(("#", "mailto:", "tel:")):
            return m.group(0)
        target, anchor = parsed.path, ("#" + parsed.fragment if parsed.fragment else "")
        if not target:
            return m.group(0)
        if ".md" in target:
            target, anchor = md_target_to_url(target, anchor)
        # Resolve against the file's OLD URL (its directory)
        absolute = posixpath.normpath(posixpath.join(old_base, target))
        is_page = target.endswith("/") or "." not in posixpath.basename(target)
        if is_page and not absolute.endswith("/"):
            absolute += "/"
        # Strip lang prefix into EN space for mapping
        en_abs = absolute[3:] if absolute.startswith("/ja/") else absolute
        ja_static_repair = False
        if absolute.startswith("/ja/") and en_abs.startswith(STATIC_PREFIXES):
            ja_static_repair = True  # static files live un-prefixed
        mapped = old2new.get(en_abs, en_abs)
        # Re-prefix pages with the language; static stays un-prefixed
        if mapped.startswith(STATIC_PREFIXES) or ja_static_repair:
            new_abs = mapped
        else:
            new_abs = (("/ja" + mapped) if is_ja else mapped)
        # Relativize from the file's NEW URL
        rel = posixpath.relpath(new_abs, new_base)
        if new_abs.endswith("/") and rel != ".":
            rel += "/"
        if rel == ".":
            rel = "./"
        new_raw = rel + anchor
        if new_raw != raw:
            changes += 1
        return "](" + new_raw + ")"

    new_text = LINK_RE.sub(sub, text)
    if changes:
        md.write_text(new_text, encoding="utf-8")
    return changes


def main():
    old2new = load_map()
    new2old = {v: k for k, v in old2new.items()}
    total = 0
    files = sorted(CONTENT.rglob("*.md"))
    if len(sys.argv) > 1:
        files = [Path(a) for a in sys.argv[1:]]
    for md in files:
        n = rewrite_file(md, old2new, new2old)
        if n:
            print(f"  {md}: {n} links rewritten")
            total += n
    print(f"Total: {total} links rewritten across {len(files)} files")


if __name__ == "__main__":
    main()
