"""
query.py

Query video "memories" stored in Pinecone using Google Vertex AI multimodal
embeddings (same model as ingest for semantic search), then re-rank results:

FinalScore = alpha * relevance + beta * importance + gamma * time_decay
"""

import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from google.oauth2 import service_account
from pinecone import Pinecone

# --------- ENV & CLIENTS ---------

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
VERTEX_AI_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_AI_PROJECT")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
QUERY_PIPELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = QUERY_PIPELINE_ROOT.parent.parent
VERTEX_AI_KEY_PATH = Path(
    os.getenv("VERTEX_AI_SERVICE_ACCOUNT_JSON", str(PROJECT_ROOT / "vertex_ai_service_account.json"))
)
EMBEDDING_DIMENSION = 1408  # must match ingest pipeline

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

pc = Pinecone(api_key=PINECONE_API_KEY)


# --------- EMBEDDING FOR TEXT QUERY (VERTEX AI MULTIMODAL, TEXT-ONLY) ---------

def get_text_embedding(text_query: str) -> List[float]:
    """
    Convert a text question into an embedding using Vertex AI multimodal model
    (text-only; same semantic space as video embeddings at ingest).
    Returns: a single 512-d vector (list of floats) to query Pinecone.
    """
    import vertexai
    from vertexai.vision_models import MultiModalEmbeddingModel

    project = VERTEX_AI_PROJECT
    credentials = None
    if VERTEX_AI_KEY_PATH.exists():
        credentials = service_account.Credentials.from_service_account_file(str(VERTEX_AI_KEY_PATH))
        if not project:
            with open(VERTEX_AI_KEY_PATH) as f:
                project = json.load(f).get("project_id")
    if not project:
        raise RuntimeError(
            "Vertex AI project not set. Set GOOGLE_CLOUD_PROJECT or VERTEX_AI_PROJECT in .env, "
            "or use a service account JSON that contains project_id."
        )

    vertexai.init(
        project=project,
        location=VERTEX_AI_LOCATION,
        credentials=credentials,
    )
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    embeddings = model.get_embeddings(
        contextual_text=text_query,
        dimension=EMBEDDING_DIMENSION,
    )
    if embeddings.text_embedding is None:
        raise RuntimeError("No text_embedding returned from Vertex AI multimodal model")
    vec = embeddings.text_embedding
    return list(vec) if hasattr(vec, "__iter__") and not isinstance(vec, str) else vec


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
    index_name: str = "memobot-memories",
    top_k: int = 10,
    alpha: float = 0.5,   # relevance weight
    beta: float = 0.3,    # importance weight
    gamma: float = 0.2,   # time-decay weight
    person_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    1) Embed the question using Vertex AI multimodal (text).
    2) Query Pinecone for similar embeddings (optionally filtered by person_id).
    3) Combine:
       - relevance (Pinecone score)
       - importance (metadata['importance_score'])
       - time (time decay from metadata['timestamp_utc'])
       into a final score and re-rank.

    person_id: Required for vector DB: only memories whose metadata person_ids
               contains this ID are returned. If None (no recognized user),
               returns no vector results.

    Returns list of dicts:
        {id, relevance_score, importance_score, time_score, final_score, metadata}
    """
    # No recognized user → no vector DB results
    if not person_id:
        return []

    # 1. Get query embedding
    query_embedding = get_text_embedding(question)

    # 2. Query Pinecone with metadata filter: only vectors for this person
    # Per https://docs.pinecone.io/guides/search/filter-by-metadata
    # person_ids is stored as a list; $in matches vectors where the field is in the given list
    index = pc.Index(index_name)
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"person_ids": {"$in": [person_id]}},
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

    # 7. Deduplicate by video_file + start_time_sec: keep only the best-scoring entry per segment
    seen = set()
    deduped = []
    for r in results:
        md = r.get("metadata") or {}
        key = (md.get("video_file"), md.get("start_time_sec"))
        if key not in seen:
            deduped.append(r)
            if key[0] is not None:
                seen.add(key)
    return deduped[:top_k]


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
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Query video memories (Pinecone + Vertex AI).")
    parser.add_argument("--person-id", type=str, default=None, help="Person ID for filtering memories (required for results).")
    parser.add_argument("question", nargs="*", help="Question to search for (or leave empty to be prompted).")
    args = parser.parse_args()

    question = " ".join(args.question).strip() if args.question else input("Enter your question: ")
    results = retrieve_and_rank(question, person_id=args.person_id)
    if not results:
        print("No matches found.")
    else:
        pretty_print_results(question, results)
