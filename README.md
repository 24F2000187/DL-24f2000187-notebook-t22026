# Smart MCQ Solver Challenge (tentative title)

**Course:** Introduction to DL and GenAI [BSDA2001P]
**Name:** VARUN CHOUDHARY
**Roll No:** 24f2000187

## Project Overview

An end-to-end deep learning / GenAI project that ranks multiple-choice-question
answers by plausibility. Given a question `prompt` and five answer choices
(`A`-`E`), the model predicts the **top three most probable answers** for each
question in the Kaggle competition test set (scored by mAP@3).

Every milestone notebook below is **actually executed, not just written** --
outputs, plots, and numbers in this README come from real runs on this
machine (CPU-only; see the compute note in each milestone).

## Dataset

| File | Description |
|---|---|
| `data/train.csv` | Training questions with the correct `answer` label (A-E) |
| `data/test.csv` | Test questions without the `answer` column |
| `data/sample_submission.csv` | Required submission format |

Columns: `id`, `prompt`, `A`, `B`, `C`, `D`, `E`, and `answer` (train only).

Raw CSVs are **not committed** to this repo (see `.gitignore`) — download them
via the Kaggle competition page or `kagglehub` and place them in `data/`
before running any script or notebook. `data/rag_corpus/` *is* committed —
it's a small (~7MB) pre-built vector database (see Milestone 3), not raw
competition data.

## Repository Structure

```
.
├── notebooks/
│   ├── EDA.ipynb                    # exploratory data analysis
│   ├── Feature_Engineering.ipynb    # feature engineering experiments
│   ├── kaggle_notebook.ipynb        # notebook submitted on Kaggle
│   ├── milestone-1.ipynb            # NLP foundation & semantic similarity
│   ├── milestone-2.ipynb            # transformers, attention, zero-shot
│   ├── milestone-3.ipynb            # RAG pipeline
│   ├── milestone-4.ipynb            # MCQ formatting + LoRA fine-tuning
│   └── milestone-5.ipynb            # ensembling
│
├── src/
│   ├── train.py             # trains the baseline TF-IDF+LogReg model
│   ├── inference.py         # loads a trained model, writes a submission.csv
│   ├── utils.py             # baseline feature engineering helpers
│   ├── text_processing.py   # text cleaning / tokenization / missing-data handling
│   ├── word2vec.py          # skip-gram Word2Vec from scratch (NumPy)
│   ├── metrics.py           # mAP@3 (this competition's metric)
│   ├── embeddings.py        # pretrained-transformer embedding extraction
│   ├── rag.py                # Wikipedia retrieval + FAISS vector store helpers
│   ├── build_rag_corpus.py  # one-time build of the milestone-3 vector database
│   ├── mcq_formatting.py    # question+option tokenization for multiple-choice models
│   └── ensembling.py        # score averaging / Borda / RRF rank fusion
│
├── reports/                 # milestone plots + reports
├── models/                  # trained model artifacts (baseline.joblib gitignored; LoRA adapter committed)
├── data/rag_corpus/         # pre-built vector database for milestone 3 (committed)
│
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place `train.csv`, `test.csv`, `sample_submission.csv` in `data/`.

## Baseline Model

A TF-IDF + Logistic Regression baseline ranks each of the 5 options per
question by predicted "correctness" probability and submits the top 3.

Train:
```bash
python src/train.py --train data/train.csv --model-out models/baseline.joblib
```

Run inference / generate a submission:
```bash
python src/inference.py --test data/test.csv --model models/baseline.joblib --out reports/submission.csv
```

## Milestones

Each milestone lives on its own `milestone-N` branch (never deleted) and is
merged into `main` once complete. Every notebook was executed end-to-end on
this CPU-only machine — no GPU available locally, so milestones 2-5
deliberately use small models/subsamples, with explicit notes on what to
change to scale up on a GPU (Kaggle/Colab).

| # | Notebook | Covers | Headline result |
|---|---|---|---|
| 1 | `milestone-1.ipynb` | Text cleaning/tokenization/missing-data handling; TF-IDF + from-scratch Word2Vec (`gensim` has no Python 3.14 wheel); prompt/option cosine similarity; mAP@3 | TF-IDF mAP@3 = 0.296 (*below* random guessing, 0.367 — this dataset's distractors share heavy surface vocabulary); Word2Vec = 0.377 |
| 2 | `milestone-2.ipynb` | HF `datasets`/`transformers`; BERT/RoBERTa architecture + real attention-heatmap visualization; pretrained DistilBERT embeddings; zero-shot NLI classification | DistilBERT cosine similarity mAP@3 = 0.401 vs. TF-IDF 0.294 on the same sample; zero-shot NLI = 0.440 |
| 3 | `milestone-3.ipynb` | LLM limitations & RAG; a pre-built FAISS vector database (176 Wikipedia pages, 1087 passages, `data/rag_corpus/`); retrieval + context-augmented zero-shot | Retrieved context *hurt* mAP@3 here (0.440 → 0.423) — traced to a concrete retrieval-quality failure, reported honestly |
| 4 | `milestone-4.ipynb` | MCQ data formatting for `AutoModelForMultipleChoice`; LoRA vs. full fine-tuning (real parameter counts); `TrainingArguments` efficiency knobs; a real LoRA fine-tune | mAP@3 improved 0.3375 (untrained head) → 0.5208 after fine-tuning on just 320 examples, 2 epochs |
| 5 | `milestone-5.ipynb` | Logit extraction/sorting; ensembling (average/weighted/Borda/RRF/stacking) across all 5 models above; a confidence-gated hybrid | The single best model (baseline LogReg, 0.594) beat every 5-model ensemble (best: 0.5625) on the shared 80-question eval set — reported as-is, not cherry-picked |

Full context, code, and discussion for each result is in the corresponding
notebook — the table above is a summary, not a replacement for reading them.

## Git Workflow (Mandatory)

This repo follows the course-mandated branching strategy:

- **`main`** always holds the latest working notebook, final training/inference
  scripts, and updated reports.
- **`milestone-N`** branches hold all work-in-progress for a given milestone
  and are merged into `main` once complete. **Milestone branches are never
  deleted**, even after merging.

```bash
git checkout main
git pull origin main
git checkout -b milestone-6
# ...work, commit...
git push origin milestone-6
# once complete:
git checkout main
git merge milestone-6
git push origin main
```

Commit messages follow `Milestone N: <what was done>` (e.g.
`Milestone 1: Completed data preprocessing and EDA`).

## Experiment Tracking

Experiments are tracked on Weights & Biases under project
`24f2000187-t22026`.

## Links

- Kaggle notebook: `DL-24f2000187-notebook-t22026`
- W&B project: `24f2000187-t22026`
- GitHub repo: (add link once created)
