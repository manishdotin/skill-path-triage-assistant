"""Frequent skill-set and chronological transition mining."""
from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations

def mine_frequent_itemsets(sequences: list[dict], min_support: float = 0.08,
                           max_size: int = 3) -> list[dict]:
    """Apriori over each employee's skill set (one transaction per employee)."""
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")
    transactions = [set(row["skills"]) for row in sequences]
    n = len(transactions)
    if not n:
        return []
    counts = Counter(skill for basket in transactions for skill in basket)
    current = {frozenset([s]): c for s, c in counts.items() if c / n >= min_support}
    found = []
    for size in range(1, max_size + 1):
        if not current:
            break
        for itemset, count in current.items():
            found.append({"items": sorted(itemset), "support": count / n, "count": count})
        if size == max_size:
            break
        previous = list(current)
        candidates = set()
        for a, b in combinations(previous, 2):
            union = a | b
            if len(union) == size + 1 and all(
                frozenset(sub) in current
                for sub in combinations(union, size)
            ):
                candidates.add(union)
        next_counts = {}
        for candidate in candidates:
            count = sum(candidate.issubset(basket) for basket in transactions)
            if count / n >= min_support:
                next_counts[candidate] = count
        current = next_counts
    return sorted(found, key=lambda x: (-x["support"], len(x["items"]), x["items"]))

def mine_transitions(sequences: list[dict], min_count: int = 2) -> list[dict]:
    """Count adjacent chronological skill transitions (A -> B)."""
    counts = Counter()
    for row in sequences:
        skills = row["skills"]
        counts.update(zip(skills, skills[1:]))
    total = sum(counts.values()) or 1
    return [
        {"from_skill": a, "to_skill": b, "count": count, "share": count / total}
        for (a, b), count in counts.most_common()
        if count >= min_count and a != b
    ]

def save_mining(itemsets: list[dict], transitions: list[dict], path="data/mining.json"):
    import json
    from pathlib import Path
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"itemsets": itemsets, "transitions": transitions},
                            indent=2), encoding="utf-8")
