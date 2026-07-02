#!/usr/bin/env python3
"""Inject Hugo `aliases` frontmatter from scripts/url_map.csv.

For every row old,new: find the content file now serving `new`
(`content/<new>.md` or `content/<new>/_index.md`), and add `old` to its
`aliases` TOML frontmatter — in the EN file AND the .ja.md sibling when one
exists (aliases are per-language; the same string in the JA sibling produces
the /ja/-prefixed stub).

Idempotent: an alias already present is not duplicated.
"""

import csv
import re
from pathlib import Path

CONTENT = Path("content")
URL_MAP_CSV = Path("scripts/url_map.csv")


def content_file_for(url: str):
    rel = url.strip("/")
    if not rel:
        return None
    single = CONTENT / (rel + ".md")
    bundle = CONTENT / rel / "_index.md"
    if single.exists():
        return single
    if bundle.exists():
        return bundle
    return None


def add_alias(md: Path, alias: str) -> bool:
    text = md.read_text(encoding="utf-8")
    m = re.match(r"^\+\+\+\n(.*?)\n\+\+\+", text, re.DOTALL)
    if not m:
        print(f"  !! no TOML frontmatter: {md}")
        return False
    fm = m.group(1)
    am = re.search(r"^aliases\s*=\s*\[(.*?)\]", fm, re.MULTILINE | re.DOTALL)
    if am:
        existing = [a.strip().strip('"').strip("'") for a in am.group(1).split(",") if a.strip()]
        if alias in existing:
            return False
        new_list = ", ".join(f'"{a}"' for a in existing + [alias])
        new_fm = fm[: am.start()] + f"aliases = [{new_list}]" + fm[am.end():]
    else:
        new_fm = fm + f'\naliases = ["{alias}"]'
    md.write_text("+++\n" + new_fm + "\n+++" + text[m.end():], encoding="utf-8")
    return True


def main():
    added, missing = 0, []
    with URL_MAP_CSV.open() as f:
        for row in csv.DictReader(f):
            old, new = row["old_path"], row["new_path"]
            en = content_file_for(new)
            if en is None:
                missing.append(new)
                continue
            targets = [en]
            ja = en.with_name(en.name.replace(".md", ".ja.md"))
            if ja.exists():
                targets.append(ja)
            for t in targets:
                if add_alias(t, old):
                    added += 1
    print(f"Aliases added: {added}")
    if missing:
        print("No content file yet for (run again after the move):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
