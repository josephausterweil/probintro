#!/usr/bin/env python3
"""Standalone runtime test for the GenJAX code blocks in Tutorial 3 Chapter 5
(Gaussian Mixture Models).

The chapter's code historically used a non-existent API. This script extracts
every ```python block from content/intro2/05_mixture_models.md and actually
EXECUTES each one against a real GenJAX install, so we can confirm the chapter
runs rather than just parses.

Usage:
    python3 test_ch5_code.py

What it does:
    1. Creates an isolated venv at .ch5_test_venv/ (reused on re-runs).
    2. pip-installs genjax (which pulls in jax) into that venv.
    3. Re-launches itself inside the venv.
    4. Extracts the python code blocks from the chapter markdown.
    5. Executes each block in a fresh namespace; reports PASS/FAIL + traceback.

Exit code 0 if every block runs, 1 otherwise.
"""

import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER = HERE / "content" / "intro2" / "05_mixture_models.md"
VENV = HERE / ".ch5_test_venv"
VENV_PYTHON = VENV / "bin" / "python"
# Marker env var so the re-launched process knows it's already inside the venv.
INSIDE_VENV_FLAG = "CH5_TEST_INSIDE_VENV"


def _find_cmake():
    """Return a Path to a usable native `cmake` binary, or None.

    Checks PATH first, then a few common conda/system locations. Excludes the
    pip `cmake` shim, which is a Python wrapper that fails inside pip's
    isolated build environments.
    """
    import shutil

    candidates = []
    on_path = shutil.which("cmake")
    if on_path:
        candidates.append(Path(on_path))
    candidates += [
        Path.home() / "anaconda3" / "bin" / "cmake",
        Path.home() / "miniconda3" / "bin" / "cmake",
        Path("/usr/bin/cmake"),
        Path("/usr/local/bin/cmake"),
    ]
    for c in candidates:
        if c.exists():
            # Verify it actually runs (the pip shim would raise here).
            try:
                probe = subprocess.run(
                    [str(c), "--version"], capture_output=True, text=True, timeout=30
                )
                if probe.returncode == 0:
                    return c
            except Exception:
                continue
    return None


def bootstrap_venv_and_relaunch():
    """Create the venv, install deps, then re-run this script inside it."""
    if not VENV_PYTHON.exists():
        print(f"[setup] Creating venv at {VENV} ...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)

    print("[setup] Upgrading pip ...")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        check=True,
    )

    # genjax depends (transitively) on an OLDER dm-tree that publishes no wheel
    # for recent Python, so pip builds it from source. That source build needs
    # a real `cmake` BINARY (the pip `cmake` shim does not survive pip's
    # isolated build environment). We locate a native cmake and put its
    # directory on PATH for the install subprocess.
    cmake_binary = _find_cmake()
    if cmake_binary is None:
        print("[setup] ERROR: no native `cmake` binary found.")
        print("[setup] genjax's dm-tree dependency needs CMake to build from source.")
        print("[setup] Install one, e.g.:  conda install -n base -y cmake")
        sys.exit(1)
    print(f"[setup] Using cmake at: {cmake_binary}")

    build_env = dict(os.environ)
    build_env["PATH"] = f"{cmake_binary.parent}{os.pathsep}{build_env.get('PATH', '')}"

    # genjax pulls jax/jaxlib as dependencies. matplotlib is imported by the
    # chapter prose in places but the three code blocks here don't need it;
    # install it anyway so the harness is robust to future blocks.
    print("[setup] Installing genjax + jax + matplotlib (this can take a few minutes) ...")
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "genjax", "matplotlib"],
        capture_output=True,
        text=True,
        env=build_env,
    )
    if result.returncode != 0:
        print("[setup] FAILED to install genjax. pip output:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print("[setup] Install complete. Re-launching test inside the venv ...\n")
    env = dict(os.environ, **{INSIDE_VENV_FLAG: "1"})
    completed = subprocess.run([str(VENV_PYTHON), __file__], env=env)
    sys.exit(completed.returncode)


def extract_python_blocks(md_path: Path):
    """Return a list of (block_index, source) for each ```python fenced block."""
    text = md_path.read_text()
    # Match ```python ... ``` blocks (non-greedy).
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    return [(i, m.group(1)) for i, m in enumerate(pattern.finditer(text))]


def run_blocks():
    """Execute each code block in a fresh namespace. Returns True if all pass."""
    if not CHAPTER.exists():
        print(f"ERROR: chapter not found at {CHAPTER}")
        return False

    blocks = extract_python_blocks(CHAPTER)
    print(f"Found {len(blocks)} python code block(s) in {CHAPTER.name}\n")

    all_passed = True
    for idx, source in blocks:
        header = f"--- Block {idx} ---"
        print(header)
        # Show the first line so it's identifiable.
        first_real_line = next(
            (ln for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")),
            "(empty)",
        )
        print(f"  first code line: {first_real_line.strip()}")
        try:
            namespace = {"__name__": "__main__"}
            exec(compile(source, f"<block {idx}>", "exec"), namespace)
            print(f"  RESULT: PASS\n")
        except Exception:
            all_passed = False
            print(f"  RESULT: FAIL")
            print("  traceback:")
            for line in traceback.format_exc().splitlines():
                print(f"    {line}")
            print()

    return all_passed


def main():
    if os.environ.get(INSIDE_VENV_FLAG) != "1":
        bootstrap_venv_and_relaunch()
        return  # bootstrap re-launches and exits

    # We're inside the venv now.
    try:
        import genjax  # noqa: F401
        import jax  # noqa: F401
        print(f"[env] genjax {getattr(genjax, '__version__', '?')}, "
              f"jax {jax.__version__}\n")
    except ImportError as e:
        print(f"[env] ERROR importing genjax/jax inside venv: {e}")
        sys.exit(1)

    ok = run_blocks()
    if ok:
        print("=" * 60)
        print("ALL BLOCKS RAN SUCCESSFULLY")
        sys.exit(0)
    else:
        print("=" * 60)
        print("ONE OR MORE BLOCKS FAILED — see tracebacks above")
        sys.exit(1)


if __name__ == "__main__":
    main()
