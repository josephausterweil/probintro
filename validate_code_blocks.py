"""
Validate code blocks in tutorial markdown files.

Two layers of checking:

1. **Syntax** (all blocks, always): every ```python / ```jax block must parse,
   and a few JAX-style lints are reported as warnings.

2. **Execution + output check** (opt-in per file, on by default for the runnable
   tutorials): blocks are actually executed, in order, in a shared namespace per
   file (so a cell may use names defined by an earlier cell, matching how the
   chapters read). If a block is immediately followed by an ``**Output:**`` fenced
   block in the markdown, the block's captured stdout is compared against it
   (exact match, with numeric tolerance so harmless float-formatting differences
   pass).

   This is what catches the failure modes a syntax-only check misses: a cell that
   raises at runtime (e.g. a dtype error), or a cell whose printed numbers no
   longer match the ``**Output:**`` the prose relies on.

Per-block control via an HTML comment on the line *immediately before* the fence:

    <!-- validate: skip -->        do not execute this block (still syntax-checked)
    <!-- validate: skip-output -->  execute, but do not compare stdout
    <!-- validate: reset -->        run this block in a FRESH namespace (drop prior state)
    <!-- validate: tol=0.05 -->     compare numbers in this block's output with an
                                    absolute tolerance of 0.05 (for stochastic
                                    Monte-Carlo cells whose estimate wobbles between
                                    runs but stays near its reference value)

Use ``skip`` for illustrative fragments / Colab-only snippets that cannot run
standalone. Every skip is reported, so nothing is silently uncovered.

Usage:
    python validate_code_blocks.py                # syntax everywhere; execute the runnable set
    python validate_code_blocks.py --no-exec      # syntax only (the old behavior)
    python validate_code_blocks.py --exec-all     # also execute intro/ and glossary fragments
    python validate_code_blocks.py content/intro2/12_hierarchical_bayes.md   # one file
"""

import argparse
import ast
import io
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Which directories are executed by default. Syntax is always checked everywhere.
# These are the GenJAX-backed, runnable tutorials. intro/ and glossary.md are
# mostly illustrative fragments; include them with --exec-all.
# ---------------------------------------------------------------------------
EXEC_DIRS_DEFAULT = ["intro2", "genjax"]
EXEC_DIRS_ALL = ["intro", "intro2", "genjax"]

# Numbers in expected-output are compared with this relative tolerance so that
# harmless last-digit / formatting differences (0.661 vs 0.6610) don't fail.
FLOAT_RTOL = 1e-3
FLOAT_ATOL = 1e-6

_NUM_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")


class Block:
    def __init__(self, line_num: int, code: str, expected: Optional[str],
                 skip_exec: bool, skip_output: bool, reset: bool,
                 tol: Optional[float] = None):
        self.line_num = line_num
        self.code = code
        self.expected = expected          # text inside the **Output:** fence, or None
        self.skip_exec = skip_exec
        self.skip_output = skip_output
        self.reset = reset
        # Absolute numeric tolerance for this block's output comparison. None means
        # use the strict global default. Set via `<!-- validate: tol=0.05 -->` for
        # stochastic Monte-Carlo cells whose printed estimate wobbles between runs
        # but should stay within `tol` of its (pinned/seeded) reference value.
        self.tol = tol


def extract_blocks(markdown_path: Path) -> List[Block]:
    """Extract python/jax code blocks, their expected-output (if any), and any
    ``<!-- validate: ... -->`` directives on the line before the fence."""
    content = markdown_path.read_text(encoding="utf-8")
    blocks: List[Block] = []

    fence = re.compile(r"```(?:python|jax)\n(.*?)```", re.DOTALL)
    for m in fence.finditer(content):
        line_num = content[: m.start()].count("\n") + 1
        code = m.group(1)

        # Directives: scan the ~120 chars before the fence for a validate comment.
        preamble = content[max(0, m.start() - 120): m.start()]
        skip_exec = "validate: skip" in preamble and "skip-output" not in preamble.split("validate: skip")[-1][:12]
        # simpler, robust parse:
        skip_exec = bool(re.search(r"<!--\s*validate:\s*skip\s*-->", preamble))
        skip_output = bool(re.search(r"<!--\s*validate:\s*skip-output\s*-->", preamble))
        reset = bool(re.search(r"<!--\s*validate:\s*reset\s*-->", preamble))
        tol_m = re.search(r"<!--\s*validate:\s*tol\s*=\s*([0-9.eE+-]+)\s*-->", preamble)
        tol = float(tol_m.group(1)) if tol_m else None

        # Expected output: an **Output:** label followed by a fenced block,
        # allowing a blank line / whitespace between them.
        tail = content[m.end(): m.end() + 4000]
        expected = None
        out_m = re.match(r"\s*\*\*Output:?\*\*\s*\n+```[a-zA-Z]*\n(.*?)```", tail, re.DOTALL)
        if out_m:
            expected = out_m.group(1)

        blocks.append(Block(line_num, code, expected, skip_exec, skip_output, reset, tol))

    return blocks


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def check_jax_usage(code: str) -> List[str]:
    warnings = []
    if re.search(r"\bif\b.*:", code) and "jnp.where" not in code:
        warnings.append(
            "Consider using jnp.where() instead of Python if/else for JAX compatibility"
        )
    if "jnp." in code and "import jax.numpy as jnp" not in code:
        warnings.append("Missing import: 'import jax.numpy as jnp'")
    if "jr." in code and "import jax.random as jr" not in code:
        warnings.append("Missing import: 'import jax.random as jr'")
    if "@genjax" in code and "import genjax" not in code:
        warnings.append("Missing import: 'import genjax' or 'from genjax import generative'")
    return warnings


def _normalize(text: str) -> List[str]:
    """Split into stripped, non-empty lines for comparison."""
    return [ln.rstrip() for ln in text.strip("\n").splitlines()]


def compare_output(actual: str, expected: str, tol: Optional[float] = None) -> Tuple[bool, str]:
    """Compare captured stdout against the expected **Output:** block.

    Exact line-by-line match, except numeric tokens are compared numerically. By
    default that's a tight relative tolerance (so only float-formatting / last-digit
    differences pass). If ``tol`` is given (via ``<!-- validate: tol=0.05 -->``), it is
    used as an ABSOLUTE tolerance for that block — for stochastic Monte-Carlo cells
    whose printed estimate wobbles run-to-run but should stay near its reference.
    """
    a_lines = _normalize(actual)
    e_lines = _normalize(expected)
    if len(a_lines) != len(e_lines):
        return False, (f"line count differs (got {len(a_lines)}, expected {len(e_lines)})\n"
                       f"      --- got ---\n" + _indent(actual) +
                       f"      --- expected ---\n" + _indent(expected))
    for i, (a, e) in enumerate(zip(a_lines, e_lines)):
        if not _lines_match(a, e, tol):
            return False, (f"line {i+1} differs"
                           + (f" (tol={tol})" if tol is not None else "") + ":\n"
                           f"        got:      {a!r}\n"
                           f"        expected: {e!r}")
    return True, ""


def _lines_match(a: str, e: str, tol: Optional[float] = None) -> bool:
    """A line matches if it's identical, or identical once numbers are compared
    numerically within tolerance (and the non-numeric skeleton is the same).

    If ``tol`` is given, each numeric pair must agree within that ABSOLUTE tolerance;
    otherwise the strict global relative/absolute defaults apply.
    """
    if a == e:
        return True
    a_nums = _NUM_RE.findall(a)
    e_nums = _NUM_RE.findall(e)
    if len(a_nums) != len(e_nums):
        return False
    # The text with numbers blanked out must match exactly.
    if _NUM_RE.sub("§", a) != _NUM_RE.sub("§", e):
        return False
    for an, en in zip(a_nums, e_nums):
        try:
            af, ef = float(an), float(en)
        except ValueError:
            if an != en:
                return False
            continue
        if tol is not None:
            if abs(af - ef) > tol:
                return False
        elif abs(af - ef) > max(FLOAT_ATOL, FLOAT_RTOL * abs(ef)):
            return False
    return True


def _indent(text: str) -> str:
    return "".join("        " + ln + "\n" for ln in text.strip("\n").splitlines())


def validate_file(markdown_path: Path, do_exec: bool,
                  namespace: Optional[dict] = None) -> Tuple[bool, dict]:
    print(f"\nValidating {markdown_path}...")
    blocks = extract_blocks(markdown_path)
    if not blocks:
        print("  No code blocks found")
        return True, {}

    print(f"  Found {len(blocks)} code blocks" + (" (executing)" if do_exec else ""))

    all_ok = True
    stats = {"syntax_fail": 0, "run_fail": 0, "output_mismatch": 0,
             "output_ok": 0, "ran_ok": 0, "skipped": 0}
    # Namespace is shared across blocks in this file. A caller may pass one in to
    # share it across the parts of a multi-file page bundle (so a cell in part 2
    # can use names defined in part 1).
    if namespace is None:
        namespace = {}

    # Force a non-interactive matplotlib backend so plotting cells run without a
    # display, and make plt.show() a no-op (cells call it but it must not block).
    if do_exec:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as _plt
            _plt.show = lambda *a, **k: None
        except Exception:
            pass  # matplotlib not installed -> plot cells will report their own error

    for b in blocks:
        ok_syntax, err = validate_python_syntax(b.code)
        if not ok_syntax:
            print(f"  ❌ Block at line {b.line_num}: {err}")
            all_ok = False
            stats["syntax_fail"] += 1
            continue

        warnings = check_jax_usage(b.code)

        if not do_exec or b.skip_exec:
            tag = "skip-exec" if b.skip_exec else "syntax-only"
            if b.skip_exec:
                stats["skipped"] += 1
            if warnings:
                print(f"  ⚠️  Block at line {b.line_num} ({tag}):")
                for w in warnings:
                    print(f"      {w}")
            else:
                print(f"  ✅ Block at line {b.line_num} ({tag})")
            continue

        if b.reset:
            namespace = {}

        # Execute in the shared namespace, capturing stdout.
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(b.code, f"{markdown_path}:{b.line_num}", "exec"), namespace)
        except Exception:
            print(f"  ❌ Block at line {b.line_num}: RUNTIME ERROR")
            tb = traceback.format_exc().strip().splitlines()
            for ln in tb[-4:]:
                print(f"      {ln}")
            all_ok = False
            stats["run_fail"] += 1
            continue

        actual = buf.getvalue()

        if b.expected is not None and not b.skip_output:
            match, detail = compare_output(actual, b.expected, b.tol)
            if match:
                print(f"  ✅ Block at line {b.line_num} (ran + output matches)")
                stats["output_ok"] += 1
            else:
                print(f"  ❌ Block at line {b.line_num}: OUTPUT MISMATCH — {detail}")
                all_ok = False
                stats["output_mismatch"] += 1
        else:
            note = "ran; output not compared" if b.skip_output else "ran; no Output block"
            if warnings:
                print(f"  ⚠️  Block at line {b.line_num} ({note}):")
                for w in warnings:
                    print(f"      {w}")
            else:
                print(f"  ✅ Block at line {b.line_num} ({note})")
            stats["ran_ok"] += 1

    return all_ok, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="specific markdown files (default: all tutorials)")
    ap.add_argument("--no-exec", action="store_true", help="syntax only; do not execute blocks")
    ap.add_argument("--exec-all", action="store_true",
                    help="also execute intro/ and glossary fragments (default skips those dirs)")
    args = ap.parse_args()

    content_dir = Path("content")
    if not content_dir.exists():
        print(f"Error: {content_dir} directory not found")
        sys.exit(1)

    exec_dirs = EXEC_DIRS_ALL if args.exec_all else EXEC_DIRS_DEFAULT

    if args.files:
        markdown_files = [Path(f) for f in args.files]
    else:
        tutorial_dirs = ["intro", "intro2", "genjax"]
        markdown_files = []
        for d in tutorial_dirs:
            p = content_dir / d
            if p.exists():
                markdown_files.extend(p.glob("**/*.md"))
        # glossary lives at content root
        gloss = content_dir / "glossary.md"
        if gloss.exists():
            markdown_files.append(gloss)

    if not markdown_files:
        print("No markdown files found")
        sys.exit(1)

    print(f"Found {len(markdown_files)} markdown files to validate")
    if not args.no_exec:
        print(f"Executing blocks in: {', '.join(exec_dirs)}  "
              f"(syntax-only elsewhere; use --exec-all to widen, --no-exec to disable)")

    # Group files by "execution unit". A page bundle (a directory containing an
    # `_index.md` alongside numbered/weighted part files) is one continuous chapter:
    # its parts run in weight order sharing a single namespace, so a continuation
    # cell in a later part can use names defined in an earlier part. Every other
    # file is its own unit.
    def weight_of(path: Path) -> int:
        m = re.search(r'^weight\s*=\s*(\d+)', path.read_text(encoding="utf-8"), re.MULTILINE)
        return int(m.group(1)) if m else 999

    units = {}  # key -> list of files
    for md in markdown_files:
        parent = md.parent
        is_bundle = (parent / "_index.md").exists() and parent.name != "content" \
            and parent.name not in ("intro", "intro2", "genjax")
        key = str(parent) if is_bundle else str(md)
        units.setdefault(key, []).append(md)

    all_ok = True
    totals = {}
    for key in sorted(units):
        files = units[key]
        # Within a bundle, _index first then parts by weight; otherwise just the file.
        files.sort(key=lambda p: (p.name != "_index.md", weight_of(p), p.name))
        shared_ns: dict = {}  # one namespace per execution unit
        for md in files:
            do_exec = (not args.no_exec) and any(part in exec_dirs for part in md.parts)
            ok, stats = validate_file(md, do_exec, namespace=shared_ns if do_exec else None)
            if not ok:
                all_ok = False
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v

    print("\n" + "=" * 60)
    if totals:
        print(f"Executed: {totals.get('output_ok',0)} output-checked, "
              f"{totals.get('ran_ok',0)} ran-only, "
              f"{totals.get('skipped',0)} skip-exec  |  "
              f"Failures: {totals.get('syntax_fail',0)} syntax, "
              f"{totals.get('run_fail',0)} runtime, "
              f"{totals.get('output_mismatch',0)} output-mismatch")
    if all_ok:
        print("✅ All code blocks validated successfully")
        sys.exit(0)
    else:
        print("❌ Some code blocks have errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
