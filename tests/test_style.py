"""Comment rules Ruff cannot check: every comment fits on one line of at most 100 characters.

Explanations longer than that belong in a docstring. Ruff (W505) already limits standalone
comment and docstring lines; this also covers trailing comments and comment blocks.
"""

import io
import tokenize
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
SOURCES: list[Path] = sorted(p for folder in ("smartgrasp", "tests", "isaac") for p in (ROOT / folder).glob("*.py"))
MAX_COMMENT_CHARS: int = 100


def comments(path: Path) -> list[tuple[int, str, bool]]:
    """List the comments in a Python file.

    Args:
        path: Python source file.

    Returns:
        (line number, comment text including '#', whether the comment has the line to itself).
    """
    source = path.read_text()
    lines = source.splitlines()
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        (token.start[0], token.string, lines[token.start[0] - 1].lstrip().startswith("#"))
        for token in tokens
        if token.type == tokenize.COMMENT
    ]


def test_comments_are_single_short_lines() -> None:
    """No comment is longer than 100 characters, and no two standalone comments are adjacent."""
    problems: list[str] = []
    for path in SOURCES:
        found = comments(path)
        name = path.relative_to(ROOT)
        problems += [
            f"{name}:{line}: comment over {MAX_COMMENT_CHARS} characters"
            for line, text, _ in found
            if len(text) > MAX_COMMENT_CHARS
        ]
        standalone = [line for line, _, alone in found if alone]
        problems += [
            f"{name}:{line}: comment block (use one line or a docstring)"
            for line in standalone
            if line - 1 in standalone
        ]
    assert not problems, "\n".join(problems)
