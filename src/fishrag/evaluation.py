"""Section-level ranking and metrics shared by retrieval notebooks."""

import numpy as np


def gold_sections(question):
    return {
        (reference["source_id"], heading)
        for reference in question["relevant_sources"]
        for heading in reference["section_headings"]
    }


def rank_sections(chunk_scores, keys):
    assert len(chunk_scores) == len(keys)
    seen = set()
    ranked = []
    for index in np.argsort(-np.asarray(chunk_scores), kind="stable"):
        key = keys[int(index)]
        if key not in seen:
            seen.add(key)
            ranked.append(
                {
                    "key": key,
                    "chunk_index": int(index),
                    "score": float(chunk_scores[index]),
                }
            )
    return ranked


def metrics_at_k(ranked, gold, k):
    # Undefined for unsupported questions: never create negative relevance labels.
    assert gold and k > 0
    keys = [row["key"] for row in ranked[:k]]
    hits = gold.intersection(keys)
    first_hit = next((rank for rank, key in enumerate(keys, 1) if key in gold), None)
    return {
        "recall": len(hits) / len(gold),
        "reciprocal_rank": 1 / first_hit if first_hit is not None else 0.0,
        "all_required_found": gold.issubset(keys),
    }
