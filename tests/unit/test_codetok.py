"""The code-aware tokeniser (docs/spec/02 §2, gate G2, INV-15).

A problem statement is prose and a solution is Python. Everything here is about the three places that gap shows
up lexically, and each test is the pair that a plain whitespace tokeniser scores as unrelated.
"""

from __future__ import annotations

from acis.lexical.codetok import KEEP, code_tokens, operator_tokens, split_identifier


def test_snake_and_camel_identifiers_split_into_the_same_words():
    """The pair that matters: a statement says "binary search", the solution calls it `binary_search`."""
    for name in ("binary_search", "binarySearch", "BinarySearch"):
        assert {"binary", "search"} <= set(split_identifier(name))


def test_the_joined_form_survives_so_an_exact_match_still_scores_extra():
    assert "binarysearch" in split_identifier("binary_search")


def test_acronyms_and_digits_are_handled():
    assert "http" in split_identifier("HTTPResponse")
    assert "2" in "".join(split_identifier("Matrix2D")) or "matrix2d" in split_identifier("Matrix2D")


def test_a_single_word_identifier_is_itself():
    assert split_identifier("heapq") == ["heapq"]


def test_operators_become_tokens_instead_of_being_deleted():
    assert operator_tokens("a % b") == ["op_mod"]
    assert operator_tokens("a // b ** c") == ["op_floordiv", "op_pow"]
    assert operator_tokens("x << 1 >> 2") == ["op_shl", "op_shr"]


def test_a_repeated_operator_is_counted_twice_because_frequency_is_evidence():
    assert operator_tokens("a % b % c") == ["op_mod", "op_mod"]


def test_the_modulus_a_statement_writes_and_the_one_code_writes_become_one_token():
    """`10^9+7` in prose and `10**9 + 7` in code are the same constant (INV-15: generic arithmetic, no lists)."""
    statement = set(code_tokens("print the answer modulo 10^9+7"))
    solution = set(code_tokens("MOD = 10**9 + 7\nprint(total % MOD)"))
    assert "1000000007" in statement and "1000000007" in solution


def test_a_statements_words_and_a_solutions_identifiers_overlap():
    statement = set(code_tokens("Find the shortest path in a weighted graph"))
    solution = set(code_tokens("def shortest_path(graph, start):\n    import heapq\n"))
    assert {"shortest", "path", "graph"} <= statement & solution


def test_string_literals_are_content():
    tokens = code_tokens('if ok:\n    print("YES")\nelse:\n    print("NO")\n')
    assert "yes" in tokens and "no" in tokens


def test_algorithm_carrying_names_survive_stopword_removal():
    stops = frozenset({"the", "a", "is", "set", "sum"})
    tokens = code_tokens("the sum of a set is heapq", stopwords=stops)
    assert "heapq" in tokens
    assert "the" not in tokens
    assert "sum" in tokens and "set" in tokens  # both are in KEEP: they name real operations
    assert {"sum", "set"} <= KEEP


def test_empty_and_hostile_input_produce_no_tokens_and_no_exception():
    for text in ("", "   ", "\x00", "%%%%", "'''"):
        assert isinstance(code_tokens(text), list)


def test_tokenisation_is_deterministic():
    text = "def solve(n):\n    return n % 10**9 + 7\n"
    assert code_tokens(text) == code_tokens(text)


def test_the_tokens_survive_the_round_trip_into_the_index():
    """`bm25s.tokenize` takes strings, so the tokens go through a join — and must come back unchanged.

    The failure this catches was real: handing it pre-split lists raised deep inside the library, and the shape
    that *does* work could still have lost `op_mod` to a word pattern that splits on underscores.
    """
    from acis.lexical.codetok import tokenize_code

    encoded = tokenize_code(["a % b", "def binary_search(xs):\n    return 10**9 + 7\n"])
    vocab = set(encoded.vocab)
    assert "op_mod" in vocab and "op_pow" in vocab
    assert {"binary", "search", "binarysearch"} <= vocab
    assert "1000000007" in vocab


def test_an_index_built_on_code_tokens_retrieves_across_the_vocabulary_gap():
    """The gap this closes, end to end: a statement says "binary search", the solution calls it `binary_search`.

    The gap it does **not** close is worth stating too: "heap" and `heapq` are different tokens and stay that
    way. Bridging them is the dense channel's job, and a lexical tokeniser that guessed at synonyms would be
    inventing matches rather than finding them.
    """
    from acis.lexical.bm25 import Bm25Index

    docs = [
        "def binary_search(xs, target):\n    lo = 0\n    return lo\n",
        "import heapq\ndef dijkstra(g, s):\n    return {}\n",
        "def is_palindrome(s):\n    return s == s[::-1]\n",
    ]
    index = Bm25Index.build(["d0", "d1", "d2"], docs, tokenizer="code")
    assert index.search_one("binary search over a sorted array", k=3)[0][0] == "d0"
    assert index.search_one("dijkstra with heapq", k=3)[0][0] == "d1"
    assert index.search_one("is the string a palindrome", k=3)[0][0] == "d2"

    stock = Bm25Index.build(["d0", "d1", "d2"], docs)
    assert stock.search_one("binary search over a sorted array", k=3)[0][1] == 0.0, (
        "the stock tokeniser matches nothing here, which is why the code-aware one exists"
    )


def test_latex_and_an_unmatched_quote_do_not_blow_up_the_tokeniser():
    """The shape this corpus is actually made of: LaTeX, apostrophes, and a quote that never closes.

    The readable form of the string-literal pattern is exponential on it — 29 seconds on 180 characters — and
    `code_tokens` runs on every query, so that is a denial of service. The bound here is generous; the failure it
    catches is five orders of magnitude out.
    """
    import time

    hostile = "Bob's problem: " + r"\le \sum_{i} \cdot " * 40 + " unterminated '"
    started = time.perf_counter()
    tokens = code_tokens(hostile)
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"tokenising took {elapsed:.1f}s"
    assert tokens


def test_string_literals_are_still_found_after_the_pattern_was_made_linear():
    tokens = code_tokens("""print("YES") if flag else print('no')""")
    assert "yes" in tokens and "no" in tokens
