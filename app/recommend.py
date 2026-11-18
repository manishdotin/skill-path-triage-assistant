"""Transparent non-neural baseline and role-gap analysis."""
from __future__ import annotations
from collections import Counter, defaultdict
from .data import TRACKS

def fit_transition_model(sequences: list[dict]) -> dict:
    transitions = defaultdict(Counter)
    skill_counts = Counter()
    for row in sequences:
        skills = row["skills"]
        skill_counts.update(skills)
        for a, b in zip(skills, skills[1:]):
            transitions[a][b] += 1
    return {"transitions": {k: dict(v) for k, v in transitions.items()},
            "skill_counts": dict(skill_counts)}

def suggest_next(skills: list[str], model: dict, top_k: int = 5) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    transitions = model.get("transitions", {})
    votes = Counter()
    evidence = defaultdict(list)
    for skill in skills:
        for candidate, count in transitions.get(skill, {}).items():
            if candidate not in skills:
                votes[candidate] += count
                evidence[candidate].append(f"{skill} → {candidate} ({count})")
    if not votes:
        # Safe fallback: globally frequent skills not already present.
        for skill, count in model.get("skill_counts", {}).items():
            if skill not in skills:
                votes[skill] += count
    total = sum(votes.values()) or 1
    return [{"skill": name, "score": round(count / total, 4),
             "evidence": "; ".join(evidence[name]) or "common in training histories"}
            for name, count in votes.most_common(top_k)]

def skill_gaps(role: str, skills: list[str]) -> list[str]:
    if role not in TRACKS:
        raise ValueError(f"Unknown role '{role}'. Choose from: {', '.join(TRACKS)}")
    owned = set(skills)
    return [skill for skill in TRACKS[role] if skill not in owned]

def train_test_split(rows: list[dict], test_fraction: float = 0.2, seed: int = 7):
    import random
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - test_fraction)))
    return shuffled[:cut], shuffled[cut:]
