"""
Validate code blocks in tutorial markdown files.

This script extracts Python code blocks from markdown files and validates them
using the standard Python compiler and JAX imports.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

# JAX code validation patterns
JAX_IMPORTS = {
    "jax",
    "jax.numpy",
    "jax.random",
    "jax.scipy",
    "genjax",
    "genjax.generative",
    "genjax.inference",
}


def extract_code_blocks(markdown_path: Path) -> List[Tuple[int, str]]:
    """Extract Python code blocks from markdown file.

    Returns:
        List of (line_number, code) tuples
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match ```python or ```jax code blocks
    pattern = r"```(?:python|jax)\n(.*?)```"
    blocks = []

    for match in re.finditer(pattern, content, re.DOTALL):
        # Find line number of code block start
        line_num = content[: match.start()].count("\n") + 1
        code = match.group(1)
        blocks.append((line_num, code))

    return blocks


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python syntax.

    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def check_jax_usage(code: str) -> List[str]:
    """Check for common JAX usage issues.

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for Python if/else instead of jnp.where
    if re.search(r"\bif\b.*:", code) and "jnp.where" not in code:
        warnings.append(
            "Consider using jnp.where() instead of Python if/else for JAX compatibility"
        )

    # Check for missing imports of used JAX modules
    if "jnp." in code and "import jax.numpy as jnp" not in code:
        warnings.append("Missing import: 'import jax.numpy as jnp'")

    if "jr." in code and "import jax.random as jr" not in code:
        warnings.append("Missing import: 'import jax.random as jr'")

    if "@genjax" in code and "import genjax" not in code:
        warnings.append(
            "Missing import: 'import genjax' or 'from genjax import generative'"
        )

    return warnings


def validate_file(markdown_path: Path) -> bool:
    """Validate all code blocks in a markdown file.

    Returns:
        True if all blocks are valid, False otherwise
    """
    print(f"\nValidating {markdown_path}...")

    blocks = extract_code_blocks(markdown_path)
    if not blocks:
        print("  No code blocks found")
        return True

    print(f"  Found {len(blocks)} code blocks")

    all_valid = True
    for line_num, code in blocks:
        # Validate syntax
        is_valid, error = validate_python_syntax(code)
        if not is_valid:
            print(f"  ❌ Block at line {line_num}: {error}")
            all_valid = False
        else:
            # Check JAX usage
            warnings = check_jax_usage(code)
            if warnings:
                print(f"  ⚠️  Block at line {line_num}:")
                for warning in warnings:
                    print(f"      {warning}")
            else:
                print(f"  ✅ Block at line {line_num}")

    return all_valid


def main():
    """Validate all tutorial markdown files."""
    content_dir = Path("content")

    if not content_dir.exists():
        print(f"Error: {content_dir} directory not found")
        sys.exit(1)

    # Find all markdown files in tutorial directories
    tutorial_dirs = ["intro", "intro2", "genjax"]
    markdown_files = []

    for dir_name in tutorial_dirs:
        dir_path = content_dir / dir_name
        if dir_path.exists():
            markdown_files.extend(dir_path.glob("**/*.md"))

    if not markdown_files:
        print("No markdown files found")
        sys.exit(1)

    print(f"Found {len(markdown_files)} markdown files to validate")

    all_valid = True
    for md_file in sorted(markdown_files):
        if not validate_file(md_file):
            all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All code blocks validated successfully")
        sys.exit(0)
    else:
        print("❌ Some code blocks have errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
