"""Command-line interface: python -m app.cli --help."""
import argparse
import json
from pathlib import Path
from . import core
from .data import TRACKS, load_sequences
from .recommend import fit_transition_model, suggest_next, skill_gaps

def _csv(value):
    return [x.strip() for x in value.split(",") if x.strip()]

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="skill-path",
        description="Explore synthetic skill pathways, mine transitions, and suggest next learning steps."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the application starts")
    p = sub.add_parser("prepare", help="generate synthetic chronological employee skill histories")
    p.add_argument("--per-role", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("mine", help="mine frequent skill sets and adjacent transitions")
    p.add_argument("--min-support", type=float, default=0.08)
    p = sub.add_parser("train-lm", help="continue pretraining a small pretrained causal language model")
    p.add_argument("--model", default=None, help="Hugging Face model name; default: sshleifer/tiny-gpt2")
    p.add_argument("--epochs", type=int, default=3)
    p = sub.add_parser("evaluate", help="evaluate next-skill prediction on held-out synthetic employees")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--lm", action="store_true", help="evaluate saved continued-pretrained LM")
    p = sub.add_parser("suggest", help="suggest next skills from a comma-separated skill history")
    p.add_argument("--skills", required=True, help='e.g. "Python, SQL, statistics"')
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--lm", action="store_true", help="use saved continued-pretrained LM")
    p = sub.add_parser("gaps", help="compare current skills with a role pathway")
    p.add_argument("--role", required=True, choices=sorted(TRACKS))
    p.add_argument("--skills", default="", help='comma-separated skills already acquired')
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Skill Path Triage Assistant is ready.")
    elif args.command == "prepare":
        rows = core.milestone_1(n_per_role=args.per_role, seed=args.seed)
        print(f"Created {len(rows)} synthetic employee histories in data/sequences.json")
    elif args.command == "mine":
        result = core.milestone_2(min_support=args.min_support)
        print(f"Frequent itemsets: {len(result['itemsets'])}")
        print("Top itemsets:")
        for item in result["itemsets"][:12]:
            print(f"  {', '.join(item['items'])} | support={item['support']:.1%}")
        print("\nCommon adjacent transitions:")
        for item in result["transitions"][:12]:
            print(f"  {item['from_skill']} -> {item['to_skill']} | count={item['count']}")
        print("Saved: data/mining.json")
    elif args.command == "train-lm":
        print("Starting continued pretraining (first run downloads the pretrained checkpoint)...")
        print("Saved model to", core.milestone_3(model_name=args.model, epochs=args.epochs))
    elif args.command == "evaluate":
        print(json.dumps(core.milestone_4(top_k=args.k, use_lm=args.lm), indent=2))
    elif args.command == "suggest":
        skills = _csv(args.skills)
        if args.lm:
            from .lm import score_next_skill
            candidates = sorted({s for row in load_sequences() for s in row["skills"]} - set(skills))
            suggestions = score_next_skill(skills, candidates)[:args.top_k]
        else:
            rows = load_sequences()
            suggestions = suggest_next(skills, fit_transition_model(rows), args.top_k)
        print("Suggested next skills:")
        if not suggestions:
            print("  No matching transitions found. Try adding a current skill or use the role gap command.")
        for i, item in enumerate(suggestions, 1):
            print(f"  {i}. {item['skill']} (score: {item['score']})")
            if item.get("evidence"):
                print(f"     Evidence: {item['evidence']}")
    elif args.command == "gaps":
        missing = skill_gaps(args.role, _csv(args.skills))
        print(f"Skill gaps for {args.role}:")
        for skill in missing:
            print(f"  - {skill}")
        if not missing:
            print("  No gaps in this sample pathway.")

if __name__ == "__main__":
    main()
