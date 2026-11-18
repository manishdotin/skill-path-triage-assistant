"""Application orchestration for the Skill Path Triage Assistant."""
from __future__ import annotations
import json
from pathlib import Path
from .data import generate_sequences, save_sequences, load_sequences
from .mining import mine_frequent_itemsets, mine_transitions, save_mining
from .recommend import fit_transition_model, suggest_next, skill_gaps, train_test_split

def milestone_1(output="data/sequences.json", n_per_role=40, seed=7):
    rows = generate_sequences(n_per_role=n_per_role, seed=seed)
    save_sequences(rows, output)
    return rows

def milestone_2(path="data/sequences.json", min_support=0.08):
    rows = load_sequences(path)
    itemsets = mine_frequent_itemsets(rows, min_support=min_support)
    transitions = mine_transitions(rows)
    save_mining(itemsets, transitions)
    return {"itemsets": itemsets, "transitions": transitions}

def milestone_3(path="data/sequences.json", output_dir="models/skill-path-lm",
                model_name=None, epochs=3):
    from .lm import train_continued_pretraining
    return train_continued_pretraining(load_sequences(path), output_dir, model_name, epochs)

def milestone_4(path="data/sequences.json", top_k=5, use_lm=False):
    rows = load_sequences(path)
    train, test = train_test_split(rows)
    model = fit_transition_model(train)
    hits, reciprocal_ranks, evaluated = 0, [], 0
    lm = None
    if use_lm:
        from .lm import score_next_skill
        lm = score_next_skill
    for row in test:
        if len(row["skills"]) < 3:
            continue
        prefix, target = row["skills"][:-1], row["skills"][-1]
        candidates = sorted({s for r in train for s in r["skills"]} - set(prefix))
        if lm:
            ranked = lm(prefix, candidates)
        else:
            ranked = suggest_next(prefix, model, top_k=len(candidates))
        rank = next((i for i, item in enumerate(ranked, 1) if item["skill"] == target), None)
        evaluated += 1
        if rank is not None:
            reciprocal_ranks.append(1 / rank)
            hits += rank <= top_k
        else:
            reciprocal_ranks.append(0.0)
    return {"evaluated": evaluated, "hit_rate_at_k": hits / max(1, evaluated),
            "mrr": sum(reciprocal_ranks) / max(1, evaluated), "k": top_k,
            "method": "continued-pretrained LM" if lm else "transition-count baseline",
            "note": "Synthetic held-out employees; results are a development check, not HR validation."}

def milestone_5(skills: list[str], path="data/sequences.json", top_k=5):
    rows = load_sequences(path)
    model = fit_transition_model(rows)
    return suggest_next(skills, model, top_k=top_k)
