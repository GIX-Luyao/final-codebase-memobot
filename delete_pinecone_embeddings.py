#!/usr/bin/env python3
"""
Delete all embeddings from the Pinecone vector database.
Uses PINECONE_API_KEY and index name from env or default.

Run from project root:
  python delete_pinecone_embeddings.py
  python delete_pinecone_embeddings.py --index my-other-index
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent
from dotenv import load_dotenv

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from pinecone import Pinecone

DEFAULT_INDEX_NAME = "memobot-memories"


def main():
    parser = argparse.ArgumentParser(description="Delete all embeddings from Pinecone index.")
    parser.add_argument(
        "--index",
        default=os.getenv("PINECONE_INDEX", DEFAULT_INDEX_NAME),
        help=f"Pinecone index name (default: {DEFAULT_INDEX_NAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show index and stats, do not delete",
    )
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)
    index_name = args.index

    if index_name not in pc.list_indexes().names():
        print(f"Error: Index '{index_name}' does not exist.")
        sys.exit(1)

    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total = sum(ns.get("vector_count", 0) for ns in stats.get("namespaces", {}).values())
    if not stats.get("namespaces"):
        total = stats.get("total_vector_count", 0)

    print(f"Index: {index_name}")
    print(f"Total vectors: {total}")

    if args.dry_run:
        print("Dry run: no changes made.")
        return

    if total == 0:
        print("No vectors to delete.")
        return

    print("Deleting all vectors...")
    index.delete(delete_all=True)
    print("Done. All embeddings have been deleted.")


if __name__ == "__main__":
    main()
