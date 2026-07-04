#!/usr/bin/env python3
"""Link checker for the built Hugo site (public/).

Two modes:

  python3 scripts/check_links.py                 # internal-link mode (default)
      Walk every public/**/*.html page, extract href/src attributes, resolve
      relative URLs against the page's own URL, and assert every internal
      target exists in public/ (as a file, or as dir/index.html).

  python3 scripts/check_links.py --inventory scripts/baseline_urls.txt
      Compatibility mode: assert every URL in the pre-move inventory still
      resolves — either as a real page or as an alias redirect stub
      (an index.html containing http-equiv="refresh").

Exit code 0 = clean, 1 = broken links found.
"""

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

SITE_PREFIX = "/probintro/"  # baseURL path prefix baked into built HTML


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if value and name in ("href", "src"):
                self.links.append(value)


def page_url(public: Path, html_file: Path) -> str:
    """URL path of a built page, as the browser sees it (with site prefix)."""
    rel = html_file.relative_to(public)
    url = "/" + rel.as_posix()
    if url.endswith("/index.html"):
        url = url[: -len("index.html")]
    return SITE_PREFIX.rstrip("/") + url


def target_exists(public: Path, url_path: str) -> bool:
    """Does a site-absolute URL path (prefix stripped) exist in public/?"""
    rel = url_path.lstrip("/")
    if not rel:
        return (public / "index.html").exists()
    p = public / unquote(rel)
    if p.is_file():
        return True
    if (p / "index.html").is_file():
        return True
    # Hugo also emits bare .html pages in some configs
    if p.with_suffix(".html").is_file():
        return True
    return False


def check_internal_links(public: Path) -> int:
    broken = []
    pages = sorted(public.rglob("*.html"))
    for page in pages:
        base = page_url(public, page)
        parser = LinkExtractor()
        try:
            parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001 - report and continue
            broken.append((base, f"<parse error: {e}>"))
            continue
        for link in parser.links:
            parsed = urlparse(link)
            if parsed.scheme or parsed.netloc:
                continue  # external
            if link.startswith(("#", "mailto:", "tel:", "data:", "javascript:")):
                continue
            resolved = urlparse(urljoin(base, link)).path
            if not resolved.startswith(SITE_PREFIX):
                # A link that escapes the site prefix is broken by definition
                broken.append((base, link))
                continue
            site_path = resolved[len(SITE_PREFIX) - 1 :]  # keep leading /
            if not target_exists(public, site_path):
                broken.append((base, link))
    if broken:
        print(f"BROKEN internal links: {len(broken)}")
        for page, link in broken:
            print(f"  {page}  ->  {link}")
        return 1
    print(f"OK: all internal links resolve across {len(pages)} HTML files")
    return 0


def check_inventory(public: Path, inventory: Path) -> int:
    missing, stubs, real = [], 0, 0
    for line in inventory.read_text().splitlines():
        url = line.strip()
        if not url:
            continue
        target = public / url.lstrip("/") / "index.html"
        if not target.is_file():
            missing.append(url)
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        if 'http-equiv="refresh"' in text:
            stubs += 1
        else:
            real += 1
    if missing:
        print(f"MISSING pre-move URLs: {len(missing)}")
        for url in missing:
            print(f"  {url}")
        return 1
    print(f"OK: inventory intact — {real} real pages, {stubs} alias stubs")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", default="public", help="built site dir")
    ap.add_argument("--inventory", help="pre-move URL inventory file (mode 2)")
    args = ap.parse_args()
    public = Path(args.public)
    if not public.is_dir():
        print(f"Error: {public}/ not found — run hugo first")
        sys.exit(2)
    if args.inventory:
        sys.exit(check_inventory(public, Path(args.inventory)))
    sys.exit(check_internal_links(public))


if __name__ == "__main__":
    main()
