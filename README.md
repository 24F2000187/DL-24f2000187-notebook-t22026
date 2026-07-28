# Smart MCQ Solver Challenge (tentative title)

**Course:** Introduction to DL and GenAI [BSDA2001P]
**Name:** VARUN CHOUDHARY
**Roll No:** 24f2000187

## Project Overview

An end-to-end deep learning / GenAI project that ranks multiple-choice-question
answers by plausibility. Given a question `prompt` and five answer choices
(`A`-`E`), the model predicts the **top three most probable answers** for each
question in the Kaggle competition test set.

Workflow covered in this repo:
- Dataset handling, EDA and feature engineering
- Model training and evaluation
- Inference / Kaggle submission generation
- Experiment tracking with Weights & Biases
- (Optional) deployment on Hugging Face Spaces

## Dataset

| File | Description |
|---|---|
| `data/train.csv` | Training questions with the correct `answer` label (A-E) |
| `data/test.csv` | Test questions without the `answer` column |
| `data/sample_submission.csv` | Required submission format |

Columns: `id`, `prompt`, `A`, `B`, `C`, `D`, `E`, and `answer` (train only).

Raw CSVs are **not committed** to this repo (see `.gitignore`) — download them
via the Kaggle competition page or `kagglehub` and place them in `data/`
before running any script or notebook.

## Repository Structure

```
project-name/
│
├── notebooks/
│   ├── EDA.ipynb                  # exploratory data analysis
│   ├── Feature_Engineering.ipynb  # feature engineering experiments
│   └── kaggle_notebook.ipynb      # notebook submitted on Kaggle
│
├── src/
│   ├── train.py       # trains the baseline model, saves it to models/
│   ├── inference.py   # loads a trained model, writes a submission.csv
│   └── utils.py        # shared data loading / feature engineering helpers
│
├── reports/            # milestone & final reports (PDF)
│
├── models/              # trained model artifacts (gitignored, regenerate via train.py)
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
git checkout -b milestone-1
# ...work, commit...
git push origin milestone-1
# once complete:
git checkout main
git merge milestone-1
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
