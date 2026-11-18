import pytest
from app.data import generate_sequences, TRACKS
from app.mining import mine_frequent_itemsets, mine_transitions
from app.recommend import fit_transition_model, suggest_next, skill_gaps, train_test_split
from app import core

def test_synthetic_histories_are_chronological_and_repeatable():
    a = generate_sequences(2, seed=3)
    b = generate_sequences(2, seed=3)
    assert a == b
    assert len(a) == len(TRACKS) * 2
    assert all(len(r["skills"]) >= 5 for r in a)

def test_frequent_itemsets_and_transitions():
    rows = [{"skills": ["Python", "SQL", "statistics"]},
            {"skills": ["Python", "SQL", "dashboards"]},
            {"skills": ["Python", "SQL", "testing"]}]
    sets = mine_frequent_itemsets(rows, min_support=0.6, max_size=2)
    assert {"items": ["Python"], "support": 1.0, "count": 3} in sets
    assert any(x["items"] == ["Python", "SQL"] and x["support"] == 1.0 for x in sets)
    trans = mine_transitions(rows)
    assert any(x["from_skill"] == "Python" and x["to_skill"] == "SQL" for x in trans)

def test_suggestions_exclude_skills_already_owned():
    rows = [{"skills": ["Python", "SQL", "statistics"]},
            {"skills": ["Python", "SQL", "dashboards"]}]
    model = fit_transition_model(rows)
    recs = suggest_next(["Python"], model, 5)
    assert recs[0]["skill"] == "SQL"
    assert all(r["skill"] != "Python" for r in recs)

def test_role_gap_analysis():
    assert skill_gaps("Data Analyst", ["spreadsheets", "SQL"])[:2] == ["statistics", "Python"]
    with pytest.raises(ValueError):
        skill_gaps("Unknown", [])

def test_split_is_reproducible_and_disjoint():
    rows = generate_sequences(4)
    train, test = train_test_split(rows)
    assert not ({r["employee_id"] for r in train} & {r["employee_id"] for r in test})
    assert len(train) + len(test) == len(rows)

def test_evaluation_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    core.milestone_1(n_per_role=5)
    result = core.milestone_4()
    assert result["evaluated"] > 0
    assert 0 <= result["mrr"] <= 1


def test_lm_training_text_preserves_chronological_order():
    from app.lm import sequence_text
    text = sequence_text({"role": "Data Analyst", "skills": ["spreadsheets", "SQL", "Python"]})
    assert "spreadsheets -> SQL -> Python -> <END>" in text
    assert "Role: Data Analyst" in text
