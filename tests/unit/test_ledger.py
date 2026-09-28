

def test_a_row_appended_after_another_is_not_marked_dirty():
    """The refusal check exempted the ledger but the flag written into rows did not, so every second gate row of a
    batch read `dirty: true` from a clean tree — which is how a clean re-measure came to look like an integrity
    problem."""
    from acis.eval import ledger as lg

    assert not lg.tree_is_dirty(" M runs/ledger.jsonl")
    assert not lg.tree_is_dirty("M runs/ledger.jsonl")  # the first line, with its status column stripped
    assert lg.tree_is_dirty(" M runs/ledger.jsonl\n M src/acis/eval/ledger.py")
    assert lg.tree_is_dirty("?? scripts/new.py")
    assert not lg.tree_is_dirty("")
