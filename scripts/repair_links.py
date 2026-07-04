#!/usr/bin/env python3
"""Repair the file-relative links that fix_links.py mis-resolved.

fix_links.py resolved every link against the page's OLD URL. That is correct
for URL-relative links (../08_bayes_nets/) but WRONG for the file-relative
forms Hugo's render hooks resolve against the source file's directory:
`./06_glossary.md` and bare resource paths like `images/chibany.png`. Those
came out as <old-page-url>/<original-target> — a path under the page's own
old URL, where nothing can legitimately live.

Repair: for every file whose URL changed (old O != new N), any link that
resolves (against N) to an absolute path A with A.startswith(O) is corrupted.
Recover the original tail T = A[len(O):], then:
  - resource tails (contain a dot in the basename, e.g. images/x.png): restore
    the original string T verbatim — the render hook resolves it via page
    resources with a static/ fallback, exactly as at baseline;
  - page tails: the original was file-relative, so the intended target is
    join(parent(O), T); map it through url_map + legacy fixes and emit a
    URL-relative link from N.

Idempotent (repaired links no longer match the corruption pattern).
"""

import csv
import posixpath
import re
from pathlib import Path
from urllib.parse import urlparse

from fix_links import LEGACY_FIXES, LINK_RE, url_of  # same directory

CONTENT = Path("content")
URL_MAP_CSV = Path("scripts/url_map.csv")


def load_map():
    old2new = {}
    with URL_MAP_CSV.open() as f:
        for row in csv.DictReader(f):
            old2new[row["old_path"]] = row["new_path"]
    old2new.update(LEGACY_FIXES)
    return old2new


def repair_file(md: Path, old2new: dict, new2old: dict) -> int:
    is_ja = md.name.endswith(".ja.md")
    new_url_en = url_of(md)
    old_url_en = new2old.get(new_url_en, new_url_en)
    if old_url_en == new_url_en:
        return 0  # unmoved files were handled correctly
    lang = "/ja" if is_ja else ""
    new_base = lang + new_url_en
    old_prefix = lang + old_url_en

    text = md.read_text(encoding="utf-8")
    changes = 0

    def sub(m):
        nonlocal changes
        raw = m.group(1)
        parsed = urlparse(raw)
        if parsed.scheme or parsed.netloc or raw.startswith(("#", "mailto:", "tel:")):
            return m.group(0)
        target, anchor = parsed.path, ("#" + parsed.fragment if parsed.fragment else "")
        if not target:
            return m.group(0)
        absolute = posixpath.normpath(posixpath.join(new_base, target))
        if target.endswith("/") and not absolute.endswith("/"):
            absolute += "/"
        if not (absolute.startswith(old_prefix) and absolute != old_prefix):
            return m.group(0)
        tail = absolute[len(old_prefix):]
        basename = posixpath.basename(tail.rstrip("/"))
        if "." in basename:
            # resource (image etc.): restore the original file-relative form
            new_raw = tail + anchor
        else:
            # page reference: intended target was file-relative to the old dir
            tail_page = tail if tail.endswith("/") else tail + "/"
            parent = posixpath.dirname(old_url_en.rstrip("/")) or "/"
            intended = posixpath.normpath(posixpath.join(parent + "/", tail_page)) + "/"
            mapped = old2new.get(intended, intended)
            new_abs = ("/ja" + mapped) if is_ja else mapped
            rel = posixpath.relpath(new_abs, new_base)
            new_raw = (rel + "/" if rel != "." else "./") + anchor
        if new_raw != raw:
            changes += 1
            return "](" + new_raw + ")"
        return m.group(0)

    new_text = LINK_RE.sub(sub, text)
    if changes:
        md.write_text(new_text, encoding="utf-8")
    return changes


def main():
    old2new = load_map()
    new2old = {v: k for k, v in old2new.items() if k not in LEGACY_FIXES}
    total = 0
    for md in sorted(CONTENT.rglob("*.md")):
        n = repair_file(md, old2new, new2old)
        if n:
            print(f"  {md}: {n} links repaired")
            total += n
    print(f"Total repaired: {total}")


if __name__ == "__main__":
    main()
