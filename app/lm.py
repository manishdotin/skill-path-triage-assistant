"""Optional Hugging Face continued-pretraining pipeline.

The model is initialized from a pretrained causal LM and continues next-token
training on the synthetic skill histories. Install the optional ML requirements
before running `train-lm`.
"""
from __future__ import annotations
import os
from pathlib import Path

def _imports():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "LM training needs torch and transformers. Run: pip install -r requirements-ml.txt"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer

def sequence_text(row: dict) -> str:
    return f"Role: {row['role']}\nSkill path: " + " -> ".join(row["skills"]) + " -> <END>"

def train_continued_pretraining(rows: list[dict], output_dir: str = "models/skill-path-lm",
                                model_name: str | None = None, epochs: int = 3,
                                max_length: int = 128) -> str:
    """Continue causal-LM pretraining on role/skill paths; CPU-compatible but slower."""
    torch, AutoModelForCausalLM, AutoTokenizer = _imports()
    model_name = model_name or os.getenv("HF_MODEL", "sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    texts = [sequence_text(r) for r in rows]
    encoded = tokenizer(texts, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(encoded["input_ids"], encoded["attention_mask"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for input_ids, attention_mask in loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=input_ids, use_cache=False)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(out.loss.detach())
        print(f"epoch {epoch+1}/{epochs} - mean next-token loss: {total/max(1,len(loader)):.4f}")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    return str(out)

def score_next_skill(prefix: list[str], candidates: list[str], model_dir="models/skill-path-lm") -> list[dict]:
    """Rank candidate skills by average conditional token log-probability."""
    torch, AutoModelForCausalLM, AutoTokenizer = _imports()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    device = next(model.parameters()).device
    prompt = "Skill path: " + " -> ".join(prefix) + " ->"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    results = []
    with torch.no_grad():
        for candidate in candidates:
            candidate_ids = tokenizer(" " + candidate, return_tensors="pt").input_ids.to(device)
            full = torch.cat([prompt_ids, candidate_ids], dim=1)
            logits = model(full).logits[:, :-1, :]
            labels = full[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_scores = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            score = token_scores[:, prompt_ids.shape[1]-1:].mean().item()
            results.append({"skill": candidate, "score": score})
    return sorted(results, key=lambda x: x["score"], reverse=True)
