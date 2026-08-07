#!/usr/bin/env python3
"""Step 1/2 for Figure 1: validate the forest specification and cache the checked result.

Reads ``forest_spec.json`` (the one physical active transmission forest used for all
three panels), verifies that it is acyclic and that the declared descendant counts
D_i^(k) and D_j^(k) are correct, and writes the verified result to
``data/forest_check.json``. That cache is the sole input to ``02_make_figure.py``.

Run:
    python 01_generate_data.py
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "forest_spec.json"
DATA_DIR = HERE / "data"
CHECK_PATH = DATA_DIR / "forest_check.json"


def load_and_check_spec(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    spec = json.loads(raw.decode("utf-8"))
    nodes = spec["nodes"]
    node_ids = [str(node["id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Forest specification contains duplicate node identifiers.")

    node_by_id = {str(node["id"]): node for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for node_id, node in node_by_id.items():
        parent = node["parent"]
        if parent is None:
            roots.append(node_id)
        else:
            parent = str(parent)
            if parent not in node_by_id:
                raise ValueError(f"Node {node_id!r} has unknown parent {parent!r}.")
            children[parent].append(node_id)

    # A parent chain must terminate at a root, never revisit a node.
    for start in node_ids:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError(f"Cycle detected in parent chain from {start!r}.")
            seen.add(current)
            parent = node_by_id[current]["parent"]
            current = None if parent is None else str(parent)

    def active_children(node_id: str) -> list[str]:
        if not bool(node_by_id[node_id]["active"]):
            return []
        return [
            child
            for child in children.get(node_id, [])
            if bool(node_by_id[child]["active"])
        ]

    def descendants(root: str, depth: int) -> list[str]:
        if depth < 0:
            raise ValueError("Depth must be nonnegative.")
        level = [root] if bool(node_by_id[root]["active"]) else []
        for _ in range(depth):
            level = [child for parent in level for child in active_children(parent)]
        return level

    computed: dict[str, dict[str, Any]] = {}
    for selected_root, declaration in spec["selected_roots"].items():
        selected_root = str(selected_root)
        if selected_root not in node_by_id:
            raise ValueError(f"Unknown selected root {selected_root!r}.")
        expected = {
            int(depth): int(value)
            for depth, value in declaration["expected_descendant_counts"].items()
        }
        observed = {depth: len(descendants(selected_root, depth)) for depth in expected}
        if observed != expected:
            raise AssertionError(
                f"Descendant-count check failed for {selected_root}: "
                f"expected={expected}, observed={observed}"
            )
        local_nodes = [selected_root]
        queue: deque[str] = deque([selected_root])
        while queue:
            parent = queue.popleft()
            for child in active_children(parent):
                local_nodes.append(child)
                queue.append(child)
        computed[selected_root] = {
            "expected": {str(k): v for k, v in expected.items()},
            "observed": {str(k): v for k, v in observed.items()},
            "local_nodes": local_nodes,
            "check_passed": True,
        }

    return {
        "schema_version": "1.0",
        "specification": path.name,
        "specification_sha256": hashlib.sha256(raw).hexdigest(),
        "node_count": len(nodes),
        "active_node_count": sum(bool(node["active"]) for node in nodes),
        "physical_roots": roots,
        "directed_active_edges": [
            [str(node["parent"]), str(node["id"])]
            for node in nodes
            if node["parent"] is not None
            and bool(node["active"])
            and bool(node_by_id[str(node["parent"])]["active"])
        ],
        "selected_root_checks": computed,
        "all_checks_passed": True,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    check = load_and_check_spec(SPEC_PATH)
    CHECK_PATH.write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CHECK_PATH.relative_to(HERE)}; checked roots={list(check['selected_root_checks'])}.")


if __name__ == "__main__":
    main()
