# Skill Path Triage Assistant

A local command-line learning project that explores synthetic employee skill histories, mines frequent patterns, continues pretraining a small causal language model, and evaluates next-skill suggestions.

> **Privacy note:** The included dataset is entirely synthetic. Do not enter confidential employee records, performance reviews, protected characteristics, or other sensitive HR data. Suggestions are learning prompts, not hiring, promotion, or performance decisions.

## 1. Setup

Python 3.10+ is recommended (Python 3.12 is supported).

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The market-basket library dependency is retained from the supplied starter project. This project also includes a small transparent Apriori implementation in `app/mining.py` so its exact support calculations remain inspectable and tests do not depend on undocumented third-party APIs.

## 2. Generate chronological skill histories

```bash
python -m app.cli prepare
```

Creates 240 synthetic employee pathways across six roles:
Data Analyst, ML Engineer, Software Engineer, Cloud Engineer, Cybersecurity Analyst, and Product Analyst.

Outputs:
- `data/sequences.json` — one chronological skill path per synthetic employee
- `data/sequences.csv` — one row per employee/skill/time-step

## 3. Mine frequent skill patterns

```bash
python -m app.cli mine --min-support 0.08
```

- Frequent itemsets are mined with Apriori over each employee's acquired-skill set.
- Adjacent chronological skill transitions are counted separately (A → B).
- Results are saved to `data/mining.json`.

Support is the fraction of employee histories containing every skill in an itemset. A transition count records how often one skill directly follows another in the observed learning sequence.

## 4. Baseline evaluation

```bash
python -m app.cli evaluate --k 5
```

The assistant holds out employees, learns adjacent transitions from the training group, and predicts the final skill from each test employee's earlier skills. It reports Hit Rate@K and Mean Reciprocal Rank (MRR). The split is by employee, not by individual event, to reduce leakage.

## 5. Continue pretraining a language model (optional)

Install the ML dependencies:

```bash
pip install -r requirements-ml.txt
```

Then run:

```bash
python -m app.cli train-lm --epochs 3
```

The default checkpoint is `sshleifer/tiny-gpt2`, a small pretrained causal language model. The pipeline continues its next-token training objective on role-and-skill sequences and saves the fine-tuned checkpoint under `models/skill-path-lm`. The first run downloads the pretrained checkpoint, so an internet connection is required. CPU training works but may be slow. You can select another compatible Hugging Face causal LM with `--model`.

Evaluate the continued-pretrained model:

```bash
python -m app.cli evaluate --lm --k 5
```

Try LM-powered suggestions:

```bash
python -m app.cli suggest --skills "Python, SQL, statistics" --lm
```

## 6. Get suggestions and identify role gaps

Baseline suggestions:
```bash
python -m app.cli suggest --skills "Python, SQL, statistics" --top-k 5
```

Role gap analysis:
```bash
python -m app.cli gaps --role "ML Engineer" --skills "Python, SQL, statistics"
```

Use `python -m app.cli --help` for all commands.

## 7. Tests

```bash
pytest -q
```

The tests cover deterministic data generation, frequent itemsets, transitions, recommendations, role gaps, holdout evaluation, and CLI commands. The LM download/training is optional and is not run during CI.

## Method and limitations

- Skill paths are simplified synthetic examples, not evidence about real employee progression.
- The transition-count baseline is intentionally transparent and provides a benchmark for the LM.
- The language model predicts likely text continuations; its scores are not proof that a course is suitable for a particular person.
- The evaluation is a small offline next-skill prediction test. Real deployment would require consented, representative data, human review, privacy safeguards, and evaluation for unequal recommendation quality across groups.
- Course recommendations should be optional and employee-directed. Never use this prototype to make hiring, firing, promotion, compensation, or other consequential employment decisions.

## Milestones

- [x] 1. Define employee skill sets/roles as chronological sequences.
- [x] 2. Mine frequent skill sets and skill-to-skill transitions.
- [x] 3. Add a continued-pretraining pipeline initialized from a pretrained causal LM.
- [x] 4. Add held-out next-skill evaluation and a transition-count baseline.
- [x] 5. Add CLI commands for suggestions and role skill-gap analysis.

**Important:** The LM milestone is implemented as a runnable training pipeline. It becomes trained only after you install `requirements-ml.txt` and run `train-lm`; model weights are not bundled in this starter archive.
