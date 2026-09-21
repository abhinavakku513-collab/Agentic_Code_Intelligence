"""`acis.rank` — score composition today; candidates, PRF, LTR and confidence from Phase 4."""

from __future__ import annotations

from acis.rank.compose import assert_mode_a_contract, order_with_duplicate_tiebreak, rank_derived_scores

__all__ = ["assert_mode_a_contract", "order_with_duplicate_tiebreak", "rank_derived_scores"]
