"""Synthetic, privacy-safe employee learning histories."""
from __future__ import annotations
import csv
import json
import random
from pathlib import Path

TRACKS = {
    "Data Analyst": ["spreadsheets", "SQL", "statistics", "Python", "data cleaning",
                     "data visualization", "dashboards", "communication", "experimentation"],
    "ML Engineer": ["Python", "SQL", "statistics", "machine learning", "feature engineering",
                    "model evaluation", "deep learning", "deployment", "MLOps"],
    "Software Engineer": ["programming fundamentals", "Git", "data structures", "testing",
                          "APIs", "databases", "system design", "cloud", "CI/CD"],
    "Cloud Engineer": ["Linux", "networking", "scripting", "cloud fundamentals", "containers",
                       "infrastructure as code", "monitoring", "security basics", "CI/CD"],
    "Cybersecurity Analyst": ["networking", "Linux", "security basics", "log analysis",
                              "incident response", "vulnerability management", "Python",
                              "cloud security", "risk assessment"],
    "Product Analyst": ["spreadsheets", "SQL", "statistics", "experimentation",
                        "product metrics", "data visualization", "storytelling",
                        "stakeholder management", "roadmapping"],
}

def generate_sequences(n_per_role: int = 40, seed: int = 7) -> list[dict]:
    """Create chronological synthetic skill histories; never represents real employees."""
    rng = random.Random(seed)
    rows = []
    employee_no = 1
    for role, canonical in TRACKS.items():
        for _ in range(n_per_role):
            # Keep chronology but vary the path by skipping one optional step and
            # occasionally inserting a transferable foundation skill.
            seq = canonical[:]
            if rng.random() < 0.35 and len(seq) > 6:
                seq.pop(rng.randrange(2, len(seq)-1))
            if rng.random() < 0.30:
                seq.insert(rng.randrange(1, len(seq)), "communication")
            rows.append({
                "employee_id": f"SYN-{employee_no:04d}",
                "role": role,
                "skills": seq,
            })
            employee_no += 1
    return rows

def save_sequences(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (path.parent / "sequences.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["employee_id", "role", "step", "skill"])
        writer.writeheader()
        for row in rows:
            for step, skill in enumerate(row["skills"], 1):
                writer.writerow({"employee_id": row["employee_id"], "role": row["role"],
                                 "step": step, "skill": skill})

def load_sequences(path: str | Path = "data/sequences.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
