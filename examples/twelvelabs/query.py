"""
query.py

Query video "memories" stored in Pinecone using TwelveLabs embeddings,
then re-rank results with an AI-Town style score:

FinalScore = alpha * relevance + beta * importance + gamma * time_decay
"""

import os
import math
from datetime import datetime, timezone
from typing import List, Dict, Any

from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from pinecone import Pinecone

# --------- ENV & CLIENTS ---------

load_dotenv()
TL_API_KEY = os.getenv("TWELVE_LABS_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not TL_API_KEY:
    raise RuntimeError("TWELVE_LABS_API_KEY not set in environment")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

twelvelabs_client = TwelveLabs(api_key=TL_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# --------- EMBEDDING FOR TEXT QUERY (USING TWELVELABS) ---------

def get_text_embedding(text_query: str) -> List[float]:
    """
    Convert a text question into an embedding using TwelveLabs Embed API.

    Matches the pattern:

        res = client.embed.create(model_name="marengo3.0", text="...")
        res.text_embedding.segments[0].float_

    Returns: a single vector (list of floats) to query Pinecone.
    """
    res = twelvelabs_client.embed.create(
        model_name="marengo3.0",
        text=text_query,
    )

    if res.text_embedding is None or res.text_embedding.segments is None:
        raise RuntimeError("No text_embedding segments returned from TwelveLabs")

    segments = res.text_embedding.segments
    if len(segments) == 0:
        raise RuntimeError("Empty text_embedding.segments returned from TwelveLabs")

    # Use the first segment as the query embedding
    return segments[0].float_


# --------- TIME DECAY & FINAL SCORE ---------

def time_decay_score(
    timestamp_utc: str,
    now: datetime = None,
    half_life_hours: float = 24.0,
) -> float:
    """
    Exponential decay based on how old the memory is.

    timestamp_utc: ISO format string (e.g. '2025-12-04T21:15:32.123456+00:00')
    half_life_hours: after this many hours, score drops to 0.5
    """
    if not timestamp_utc:
        return 1.0  # if missing, treat as "no decay"

    if now is None:
        now = datetime.now(timezone.utc)

    try:
        t = datetime.fromisoformat(timestamp_utc)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except Exception:
        return 1.0

    dt_hours = (now - t).total_seconds() / 3600.0
    if dt_hours < 0:
        return 1.0

    return math.exp(-math.log(2) * dt_hours / half_life_hours)


def normalize_scores(values: List[float]) -> List[float]:
    """
    Simple min-max normalization to [0, 1].
    If all values are equal or list is empty, returns 0.5 for all.
    """
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-9:
        return [0.5 for _ in values]

    return [(v - v_min) / (v_max - v_min) for v in values]


# --------- MAIN RETRIEVAL FUNCTION ---------

def retrieve_and_rank(
    question: str,
    index_name: str = "twelve-labs",
    top_k: int = 10,
    alpha: float = 0.5,   # relevance weight
    beta: float = 0.3,    # importance weight
    gamma: float = 0.2,   # time-decay weight
) -> List[Dict[str, Any]]:
    """
    1) Embed the question using TwelveLabs.
    2) Query Pinecone for similar embeddings.
    3) Combine:
       - relevance (Pinecone score)
       - importance (metadata['importance_score'])
       - time (time decay from metadata['timestamp_utc'])
       into a final score and re-rank.

    Returns list of dicts:
        {id, relevance_score, importance_score, time_score, final_score, metadata}
    """
    # 1. Get query embedding
    query_embedding = get_text_embedding(question)

    # 2. Query Pinecone
    index = pc.Index(index_name)
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    # Support both obj-style and dict-style access
    matches = getattr(res, "matches", None)
    if matches is None:
        matches = res.get("matches", [])

    if not matches:
        return []

    # 3. Extract raw components
    relevance_scores = [m["score"] for m in matches]

    importance_raw = []
    time_scores_raw = []
    now = datetime.now(timezone.utc)

    for m in matches:
        md = m.get("metadata", {}) or {}

        # Importance
        imp = md.get("importance_score", None)
        if imp is None:
            imp_norm = 0.5
        else:
            try:
                imp_val = float(imp)
                imp_norm = max(0.0, min(1.0, imp_val / 10.0))
            except Exception:
                imp_norm = 0.5
        importance_raw.append(imp_norm)

        # Time decay
        ts = md.get("timestamp_utc")
        t_score = time_decay_score(ts, now=now, half_life_hours=24.0)
        time_scores_raw.append(t_score)

    # 4. Normalize relevance scores to [0,1]
    relevance_norm = normalize_scores(relevance_scores)

    # 5. Combine into final score
    results = []
    for m, rel, imp, t in zip(matches, relevance_norm, importance_raw, time_scores_raw):
        final_score = alpha * rel + beta * imp + gamma * t
        results.append(
            {
                "id": m["id"],
                "relevance_score": rel,
                "importance_score": imp,
                "time_score": t,
                "final_score": final_score,
                "metadata": m.get("metadata", {}),
            }
        )

    # 6. Sort by final_score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# --------- CLI ENTRYPOINT ---------

def pretty_print_results(question: str, results: List[Dict[str, Any]], max_print: int = 5):
    print(f"\nQuestion: {question}")
    print(f"Top {min(max_print, len(results))} re-ranked memories:\n")

    for i, r in enumerate(results[:max_print], start=1):
        md = r["metadata"] or {}
        summary = md.get("summary", "(no summary)")
        start_t = md.get("start_time_sec", md.get("start_time"))
        end_t = md.get("end_time_sec", md.get("end_time"))
        timestamp = md.get("timestamp_utc", "(no timestamp)")
        importance = md.get("importance_score", "N/A")

        print(f"{i}. ID: {r['id']}")
        print(
            f"   FinalScore: {r['final_score']:.4f} "
            f"(rel={r['relevance_score']:.3f}, imp={r['importance_score']:.3f}, time={r['time_score']:.3f})"
        )
        print(f"   Video: {md.get('video_file', '(unknown)')}  segment={md.get('video_segment', '(?)')}")
        print(f"   Time range: {start_t} - {end_t} sec")
        print(f"   Ingestion timestamp: {timestamp}")
        print(f"   Importance (raw): {importance}")
        print(f"   Summary: {summary}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")

    results = retrieve_and_rank(question)
    if not results:
        print("No matches found.")
    else:
        pretty_print_results(question, results)
