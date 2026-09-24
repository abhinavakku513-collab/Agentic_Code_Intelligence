"""`acis search`: the free-text surface a judge types into (docs/spec/06 §3, INV-1, INV-10).

The command is thin by contract, so what is tested here is what a thin command can still get wrong: printing
evidence that was not re-read from the content store, hiding that a stand-in encoder served the query, or
returning a different number of results than it promised.
"""

from __future__ import annotations

import contextlib
import io
import json

import pytest

from acis.appsdata import apps
from acis.cli import main

pytestmark = pytest.mark.skipif(not apps.is_available(), reason="dataset assets are not fetched")

QUERY = "count the number of distinct substrings of a string"


@pytest.fixture(scope="module")
def payload():
    """One snapshot build, shared: 8,765 documents is not something to do once per assertion."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(["search", QUERY, "--top-k", "5", "--json", "--config", "configs/dev-standin.yaml"])
    assert code == 0
    return json.loads(buffer.getvalue())


def test_it_returns_exactly_what_it_was_asked_for(payload):
    assert len(payload["results"]) == 5
    assert [h["rank"] for h in payload["results"]] == [1, 2, 3, 4, 5]
    scores = [h["score"] for h in payload["results"]]
    assert scores == sorted(scores, reverse=True)


def test_every_result_carries_evidence_re_read_from_the_store(payload):
    """INV-1: the text shown is the stored document, byte for byte — nothing is generated or paraphrased."""
    from acis.core.hashing import sha256_text
    from acis.prep.normalize import d1

    for hit in payload["results"]:
        assert hit["source"]
        assert sha256_text(d1(hit["source"])) == hit["body_hash"]


def test_the_stand_in_encoder_is_labelled_rather_than_passed_off(payload):
    """A number produced by the harness stand-in must never look like a number produced by a model."""
    assert payload["encoder"]["submission_capable"] is False


def test_the_snapshot_and_route_are_reported(payload):
    assert payload["snapshot"]["n_units"] == len(apps.corpus_ids())
    assert payload["route"] in ("generic", "statement_like")


def test_an_empty_query_is_a_typed_error_not_a_traceback(capsys):
    assert main(["search", "   "]) == 1
    assert "invalid_input" in capsys.readouterr().err
