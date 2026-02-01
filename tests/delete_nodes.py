#!/usr/bin/env python3
"""
Delete all Graphiti nodes + relationships for a specific group_id in Neo4j.
只看数量不删除: python3 delete_nodes.py --group-id tenant_001 --dry-run
执行删除: python3 delete_nodes.py --group-id tenant_001 --yes

Flow:
1) Initialize MemobotService (from env) to verify connectivity / indices.
2) Count nodes + relationships scoped to group_id.
3) Optionally delete (requires --yes unless --dry-run).

Required env:
- NEO4J_URI
- NEO4J_USER
- NEO4J_PASSWORD
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

MEMOBOT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = MEMOBOT_ROOT / ".env"

# Parse .env ONLY (do NOT rely on external process env; do NOT provide defaults)
_DOTENV = dotenv_values(DOTENV_PATH)


def _env_required(name: str) -> str:
    """
    Read required env var from .env ONLY.
    - No defaults
    - If missing/blank -> raise error
    """
    v = _DOTENV.get(name)
    if not isinstance(v, str):
        raise RuntimeError(f"Missing required key in .env: {name}")
    v = v.strip()
    if not v:
        raise RuntimeError(f"Missing required key in .env: {name}")
    return v


async def _count_for_group(driver: Any, group_id: str) -> tuple[int, int]:
    # Relationships
    rel_records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e]-()
        WHERE e.group_id = $group_id
        RETURN count(e) AS rels
        """,
        params={"group_id": group_id},
        routing_="r",
    )
    rels = int(rel_records[0]["rels"]) if rel_records else 0

    # Nodes
    node_records, _, _ = await driver.execute_query(
        """
        MATCH (n)
        WHERE n.group_id = $group_id
        RETURN count(n) AS nodes
        """,
        params={"group_id": group_id},
        routing_="r",
    )
    nodes = int(node_records[0]["nodes"]) if node_records else 0
    return nodes, rels


async def _delete_for_group(driver: Any, group_id: str) -> None:
    # Delete relationships first (safer + faster than relying on DETACH).
    await driver.execute_query(
        """
        MATCH ()-[e]-()
        WHERE e.group_id = $group_id
        DELETE e
        """,
        params={"group_id": group_id},
    )

    # Delete nodes for that group.
    await driver.execute_query(
        """
        MATCH (n)
        WHERE n.group_id = $group_id
        DETACH DELETE n
        """,
        params={"group_id": group_id},
    )


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Delete all nodes/edges for a group_id from Neo4j (Graphiti)."
    )
    parser.add_argument("--group-id", required=True, help="Graph partition id (group_id) to delete")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts; do not delete")
    parser.add_argument("--yes", action="store_true", help="Actually perform deletion")
    args = parser.parse_args()

    # Enforce: ONLY read from .env, no defaults.
    try:
        neo4j_uri = _env_required("NEO4J_URI")
        neo4j_user = _env_required("NEO4J_USER")
        neo4j_password = _env_required("NEO4J_PASSWORD")
    except RuntimeError as e:
        print(f"[Error] {e}", file=sys.stderr)
        print(f"        Please set it in: {DOTENV_PATH}", file=sys.stderr)
        return 2

    print(f"[Info] Using Neo4j config from .env: uri={neo4j_uri!r} user={neo4j_user!r} (password hidden)")

    # If MemobotService only exposes from_env(), it will read os.environ.
    # We force os.environ to match ONLY what we parsed from .env.
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USER"] = neo4j_user
    os.environ["NEO4J_PASSWORD"] = neo4j_password

    try:
        from Memobot import MemobotService  # type: ignore
    except Exception as e:
        print(f"[Error] Failed to import MemobotService: {e}", file=sys.stderr)
        return 2

    service = MemobotService.from_env(group_id=args.group_id)
    try:
        # Per request: initialize first.
        try:
            await service.initialize()
        except Exception as e:
            print(f"[Error] Failed to initialize MemobotService / connect Neo4j: {e}", file=sys.stderr)
            print(
                "        Neo4j connection must be configured via .env ONLY:\n"
                "        - NEO4J_URI\n"
                "        - NEO4J_USER\n"
                "        - NEO4J_PASSWORD\n"
                f"        .env path: {DOTENV_PATH}",
                file=sys.stderr,
            )
            return 2

        # Use the underlying Graphiti driver (Neo4j) for direct cleanup queries.
        driver = service._builder.client.driver  # type: ignore[attr-defined]

        nodes, rels = await _count_for_group(driver, args.group_id)
        print(f"[Info] group_id={args.group_id} nodes={nodes} relationships={rels}")

        if args.dry_run or not args.yes:
            if not args.yes:
                print("[Info] Not deleting (pass --yes to delete, or --dry-run to just inspect).")
            return 0

        await _delete_for_group(driver, args.group_id)

        nodes_after, rels_after = await _count_for_group(driver, args.group_id)
        print(f"[OK] Deleted. Remaining nodes={nodes_after} relationships={rels_after}")
        return 0
    finally:
        await service.close()


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
