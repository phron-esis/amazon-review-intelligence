# CLAUDE.md

Project context for Claude Code. Place this file in the repository root.

## What this project is

A sentiment + issue classifier on Amazon customer reviews, built in PyTorch, deployed as a FastAPI service, with a **business case computed from real repeat-customer data**. It is a portfolio project for an AI/ML student with a business background. The business framing is as important as the model.

## Who I am and how I want you to work with me

I am **learning deep learning**. The purpose of this project is for me to understand PyTorch, not to produce code fast. Follow these rules:

- **Do not write large amounts of code unprompted.** Propose an approach first, let me respond.
- **Stage 2 (the LSTM in raw PyTorch) is my core learning exercise.** For that stage you are a tutor only: explain concepts, review code I wrote, help me debug. Do NOT write the model, the Dataset, the collate function, or the training loop for me.
- For other stages you may write code, but **explain what it does** and keep it readable over clever.
- When I ask "why is X broken," diagnose first and let me attempt the fix before offering one.
- If I'm about to do something that will bite me later (data leakage, wrong grouping key, etc.), say so plainly.
- Prefer asking one clarifying question over guessing at an ambiguous request.

## Dataset

**Amazon Reviews 2023** (McAuley Lab, Hugging Face: `McAuley-Lab/Amazon-Reviews-2023`), category `Grocery_and_Gourmet_Food`, 5-core subset (every user has ≥5 reviews — this is deliberate: the project needs repeat customers).

Key fields: `user_id`, `asin`, `parent_asin`, `rating`, `title`, `text`, `timestamp` (ms), `verified_purchase`, `helpful_vote`. Metadata joins on `parent_asin` and supplies `price`.

### Data rules that must never be violated

1. **Sample by `user_id`, never by row.** Row sampling destroys per-user history and silently breaks the repurchase and trajectory analyses.
2. **`timestamp` is milliseconds** — convert with `pd.to_datetime(ts, unit="ms")`.
3. **Join metadata on `parent_asin`**, not `asin` (variants share a parent).
4. **Filter `verified_purchase == True`** for the business table.
5. **Right-censoring:** a user whose last review is near the dataset end (Sep 2023) has not necessarily churned. Churn requires no verified review within 12 months AND ≥12 months of observable data after their last review.
6. **Repurchase is a proxy**, not a fact: "posted another verified-purchase review later." It underestimates true repurchase. Every derived number is a lower bound and must be labeled as such.
7. **Time-based splits only** — train on earlier reviews, test on later. No random splits. No user's reviews straddling train and test.
8. **Labels:** rating 1–2 → negative, 4–5 → positive, drop 3s. Classes are imbalanced; use per-class precision/recall/F1, never bare accuracy.

## Stage plan

1. **Data engineering** — audit, sample, modeling table, customer table, measure levers, freeze splits
2. **Baseline LSTM in raw PyTorch** (my learning rep — tutor mode only)
3. **DistilBERT fine-tune** + LSTM-vs-transformer comparison
4. **Multi-label issue tagging** (quality, shipping, price, not-as-described, taste/expiration)
5. **Deployment** — FastAPI + Docker + Streamlit dashboard
6. **Business writeup** — lever math with measured inputs

## Business levers this project must support

- **Lever 1 — churn signal:** repurchase-proxy rate after negative vs. positive reviews
- **Trajectory lever:** does a user's rating/sentiment decline across their own review sequence before they go quiet?
- **Lever 2 — issue prioritization:** which complaint categories concentrate where
- **Lever 3 — reputation upside:** deliberately left unquantified

Model quality connects to business value through precision: false-positive flags waste intervention budget. Any evaluation discussion should keep that link in view.

## Conventions

- Python 3.11, `uv` for env management, `.venv` in project root
- Source of truth is `src/`, not notebooks. Notebooks are exploration only.
- `data/` and `models/` are gitignored — never commit datasets or weights
- One commit per sub-stage; include the metric in model commit messages (e.g. `model: LSTM baseline — macro F1 = 0.78`)
- `ruff` for linting

## Where things go

- `src/data/` — Stage 1 scripts
- `src/models/` — architectures, training, evaluation
- `src/serve/` — inference + FastAPI
- `app/dashboard.py` — Streamlit
- `reports/` — audit notes, figures, business case
