"""Document features from the code itself (docs/spec/02 §2 and Appendix A, INV-5, INV-15).

Everything here is derived by **parsing**, never by running: `ast.parse` builds a tree and nothing in this module
compiles, imports, executes or `eval`s a line of it. A file that will not parse is not an error — it produces a
record with `parse_ok=False` and whatever the tolerant fallback can see, and the document stays retrievable
through the dense and lexical channels exactly as before (spec 02 §6b).

The features are the *shape* of a solution, not its topic: what it imports, how it reads input, what it prints,
which constants it carries, whether it recurses. They exist to be matched against the shape a problem statement
implies — "read t test cases", "print YES or NO", "modulo 10^9+7" — which is a signal the dense channel cannot
see and the lexical channel sees only by accident.

**INV-15**: no extractor is a closed list that gates correctness. `imports` reports whatever the file imports;
`output_literals` reports whatever it prints. Nothing here decides an answer on its own, and every value is
allowed to be absent — a missing feature is `NaN`, which LightGBM handles natively and which feature-group
dropout trains the ranker to survive.
"""

from __future__ import annotations

import ast
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

MAX_SOURCE_CHARS = 200_000
_NUMBER = re.compile(
    r"(?<![\w.])(\d{2,})(?!\.?\d)"
)  # trailing "." is punctuation, not a decimal point: "modulo 10^9+7." ends in a number
_STRING_LITERAL = re.compile(r"""(['"])((?:\\.|(?!\1).){1,40})\1""")
#: Idioms worth naming because a statement implies them: a heap, a memo table, a sieve. Reported as counts, never
#: used as a filter — a solution using none of them is an ordinary solution, not a rejected one.
IDIOM_NAMES = ("heappush", "heappop", "bisect", "deque", "defaultdict", "Counter", "lru_cache", "combinations")


@dataclass(frozen=True, slots=True)
class DocFeatures:
    """What one document looks like structurally. Every field is safe to be wrong: none of them gates a result."""

    parse_ok: bool
    n_lines: int
    n_tokens: int
    imports: frozenset[str] = frozenset()
    reads_input: int = 0
    reads_stdin: bool = False
    splits_input: bool = False
    maps_int: bool = False
    has_testcase_loop: bool = False
    output_literals: frozenset[str] = frozenset()
    numeric_constants: frozenset[int] = frozenset()
    n_functions: int = 0
    n_loops: int = 0
    max_loop_depth: int = 0
    recursive: bool = False
    memoised: bool = False
    idioms: frozenset[str] = frozenset()
    meta: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "parse_ok": self.parse_ok,
            "n_lines": self.n_lines,
            "n_tokens": self.n_tokens,
            "imports": sorted(self.imports),
            "reads_input": self.reads_input,
            "reads_stdin": self.reads_stdin,
            "splits_input": self.splits_input,
            "maps_int": self.maps_int,
            "has_testcase_loop": self.has_testcase_loop,
            "output_literals": sorted(self.output_literals),
            "numeric_constants": sorted(self.numeric_constants),
            "n_functions": self.n_functions,
            "n_loops": self.n_loops,
            "max_loop_depth": self.max_loop_depth,
            "recursive": self.recursive,
            "memoised": self.memoised,
            "idioms": sorted(self.idioms),
        }


class _Walker(ast.NodeVisitor):
    """One pass over the tree. Collects shape; understands nothing about what the program computes."""

    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.reads_input = 0
        self.reads_stdin = False
        self.splits_input = False
        self.maps_int = False
        self.output_literals: set[str] = set()
        self.numbers: set[int] = set()
        self.n_functions = 0
        self.n_loops = 0
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.memoised = False
        self.idioms: set[str] = set()
        self._function_names: set[str] = set()
        self._called: set[str] = set()

    # -- imports -------------------------------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802 — ast's own naming
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            self.imports.add(node.module.split(".")[0])
        for alias in node.names:
            if alias.name in IDIOM_NAMES:
                self.idioms.add(alias.name)
        self.generic_visit(node)

    # -- calls ---------------------------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        if name:
            self._called.add(name)
            if name in IDIOM_NAMES or name.split(".")[-1] in IDIOM_NAMES:
                self.idioms.add(name.split(".")[-1])
            if name == "input":
                self.reads_input += 1
            if name in ("sys.stdin.readline", "sys.stdin.read", "stdin.readline", "stdin.read"):
                self.reads_stdin = True
                self.reads_input += 1
            if name.endswith(".split"):
                self.splits_input = True
            if name == "map" and node.args and _call_name(node.args[0]) == "int":
                self.maps_int = True
            if name == "print":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.strip():
                        self.output_literals.add(arg.value.strip().lower()[:40])
        self.generic_visit(node)

    # -- structure -----------------------------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.n_functions += 1
        self._function_names.add(node.name)
        for decorator in node.decorator_list:
            if (_call_name(decorator) or "").split(".")[-1] in ("lru_cache", "cache"):
                self.memoised = True
                self.idioms.add("lru_cache")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._loop(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self._loop(node)

    def _loop(self, node: ast.AST) -> None:
        self.n_loops += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        """Fold a constant expression: `10**9 + 7` is the constant a statement writes as `10^9+7`.

        Evaluated structurally — `**`, `*`, `+`, `-` over integer literals, with a bound on the exponent — and
        never by `eval`: this module does not execute the file it is reading (INV-5).
        """
        value = _const_int(node)
        if value is not None and abs(value) >= 10:
            self.numbers.add(value)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, int) and not isinstance(node.value, bool) and abs(node.value) >= 10:
            self.numbers.add(int(node.value))
        elif isinstance(node.value, str) and node.value.strip() and len(node.value) <= 40:
            pass  # only `print` arguments count as output; a bare string is not evidence of what is printed
        self.generic_visit(node)

    @property
    def recursive(self) -> bool:
        return bool(self._function_names & self._called)


_MAX_FOLD_EXPONENT = 18


def _const_int(node: ast.AST) -> int | None:
    """Integer value of a constant expression, or `None` when it is not one. Never executes anything."""
    if isinstance(node, ast.Constant):
        return int(node.value) if isinstance(node.value, int) and not isinstance(node.value, bool) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_int(node.operand)
        return -inner if inner is not None else None
    if not isinstance(node, ast.BinOp):
        return None
    left, right = _const_int(node.left), _const_int(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Pow):
        return left**right if 0 <= right <= _MAX_FOLD_EXPONENT else None
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult) and abs(left) < 10**9 and abs(right) < 10**9:
        return left * right
    return None


def _call_name(node: ast.AST | None) -> str:
    """Dotted name of a call target, or `""` when it is an expression rather than a name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def extract(text: str) -> DocFeatures:
    """Features for one document. Never raises: an unparseable file yields a tolerant record (spec 02 §2)."""
    source = text[:MAX_SOURCE_CHARS]
    n_lines = source.count("\n") + 1
    n_tokens = len(source.split())
    try:
        # A corpus of other people's code is full of `SyntaxWarning` — invalid escapes, mostly. They are true
        # observations about the file and irrelevant to us: we are reading its shape, not running it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return _tolerant(source, n_lines=n_lines, n_tokens=n_tokens)

    walker = _Walker()
    try:
        walker.visit(tree)
    except RecursionError:  # a tree deep enough to exhaust the stack is still a document
        return _tolerant(source, n_lines=n_lines, n_tokens=n_tokens)

    return DocFeatures(
        parse_ok=True,
        n_lines=n_lines,
        n_tokens=n_tokens,
        imports=frozenset(walker.imports),
        reads_input=walker.reads_input,
        reads_stdin=walker.reads_stdin,
        splits_input=walker.splits_input,
        maps_int=walker.maps_int,
        has_testcase_loop=_testcase_loop(tree),
        output_literals=frozenset(walker.output_literals),
        numeric_constants=frozenset(walker.numbers),
        n_functions=walker.n_functions,
        n_loops=walker.n_loops,
        max_loop_depth=walker.max_loop_depth,
        recursive=walker.recursive,
        memoised=walker.memoised,
        idioms=frozenset(walker.idioms),
    )


def _testcase_loop(tree: ast.AST) -> bool:
    """`for _ in range(int(input()))` **and** the commoner `t = int(input())` … `for _ in range(t)`.

    Recognised structurally — a loop whose iteration count came from input, directly or through one assignment —
    rather than by matching a known spelling, so an unfamiliar variant is still recognised and an unrecognised
    one costs nothing (INV-15). One level of indirection is deliberate: chasing further would need real dataflow
    for a feature that is a hint.
    """
    from_input: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _reads_input(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    from_input.add(target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if _reads_input(node.iter):
            return True
        for inner in ast.walk(node.iter):
            if isinstance(inner, ast.Name) and inner.id in from_input:
                return True
    return False


def _reads_input(node: ast.AST | None) -> bool:
    """True when this expression pulls a value from standard input, however it is spelled."""
    if node is None:
        return False
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            name = _call_name(inner.func)
            if name in ("input", "sys.stdin.readline", "stdin.readline", "sys.stdin.read", "stdin.read"):
                return True
    return False


def _tolerant(source: str, *, n_lines: int, n_tokens: int) -> DocFeatures:
    """What can be seen without a parse: literals and numbers by regex. Honest about being a fallback."""
    return DocFeatures(
        parse_ok=False,
        n_lines=n_lines,
        n_tokens=n_tokens,
        output_literals=frozenset(
            m.group(2).strip().lower()[:40] for m in _STRING_LITERAL.finditer(source) if m.group(2).strip()
        ),
        numeric_constants=frozenset(int(m.group(1)) for m in _NUMBER.finditer(source) if len(m.group(1)) <= 18),
        reads_input=source.count("input("),
        reads_stdin="stdin" in source,
    )


def extract_many(texts: Iterable[str]) -> list[DocFeatures]:
    return [extract(t) for t in texts]


__all__ = ["IDIOM_NAMES", "MAX_SOURCE_CHARS", "DocFeatures", "extract", "extract_many"]
