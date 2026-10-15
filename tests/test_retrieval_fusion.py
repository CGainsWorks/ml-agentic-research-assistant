from __future__ import annotations

from araa.retrieval.bm25 import tokenize
from araa.retrieval.hybrid import reciprocal_rank_fusion
from evals.harness import hit_at_k, mrr, ndcg_at_k, recall_at_k, wilson_interval


def test_rrf_rewards_agreement_between_retrievers():
    vector = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("c", 12.0), ("a", 9.0), ("d", 3.0)]
    fused = reciprocal_rank_fusion([vector, bm25], k=60)
    ranked = sorted(fused, key=lambda cid: -fused[cid])
    # 'a' is 1st and 2nd; 'c' is 3rd and 1st. Both beat single-list-only items.
    assert set(ranked[:2]) == {"a", "c"}
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["d"]


def test_rrf_ignores_score_scale():
    """The whole reason for using RRF: BM25 scores are unbounded and cosine
    similarities are not, so no fixed weighting between them is stable."""
    vector = [("a", 0.9), ("b", 0.1)]
    bm25_small = [("b", 0.4), ("a", 0.2)]
    bm25_huge = [("b", 4000.0), ("a", 2000.0)]
    assert reciprocal_rank_fusion([vector, bm25_small], 60) == reciprocal_rank_fusion(
        [vector, bm25_huge], 60
    )


def test_rrf_k_damps_top_rank_dominance():
    lists = [[("a", 1.0), ("b", 0.5)]]
    small_k = reciprocal_rank_fusion(lists, k=1)
    large_k = reciprocal_rank_fusion(lists, k=1000)
    assert small_k["a"] / small_k["b"] > large_k["a"] / large_k["b"]


def test_tokenizer_splits_compounds_for_partial_matching():
    tokens = tokenize("Self-attention in BERT-large achieves 88.5 F1.")
    assert "self-attention" in tokens
    assert "attention" in tokens
    assert "bert-large" in tokens
    assert "bert" in tokens


def test_tokenizer_keeps_technical_terms_that_look_like_stopwords():
    assert "not" in tokenize("The NOT gate and the no-op baseline")


def test_metrics_are_sane():
    retrieved = ["x", "gold1", "y", "gold2"]
    gold = {"gold1", "gold2"}
    assert hit_at_k(retrieved, gold, 1) == 0.0
    assert hit_at_k(retrieved, gold, 2) == 1.0
    assert recall_at_k(retrieved, gold, 4) == 1.0
    assert recall_at_k(retrieved, gold, 2) == 0.5
    assert mrr(retrieved, gold) == 0.5
    # ranking gold higher must score better
    assert ndcg_at_k(["gold1", "gold2", "x"], gold, 3) > ndcg_at_k(retrieved, gold, 4)


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_wilson_interval_widens_on_small_samples():
    lo_small, hi_small = wilson_interval(8, 10)
    lo_big, hi_big = wilson_interval(800, 1000)
    assert (hi_small - lo_small) > (hi_big - lo_big)
    assert 0.0 <= lo_small <= 0.8 <= hi_small <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
