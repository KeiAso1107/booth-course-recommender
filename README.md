# Chicago Booth Course Recommender

AI-powered course selection, bid prediction, and degree tracking tool for Chicago Booth MBA students (2025–26 Academic Year).

## Features

| Tab | Description | AI/ML Technique |
|-----|-------------|-----------------|
| **Course Explorer** | Search and filter 400+ sections by keyword, schedule, requirements, or natural language | Claude Haiku API (NL → structured filters) |
| **Bid Predictor** | Predict clearing prices and optimize bid allocation across courses | Two-stage Random Forest (classifier + regressor); scipy SLSQP optimization |
| **Degree Tracker** | Track Foundation, FLMBE, concentration progress with smart recommendations | Rule-based scoring with requirement gap analysis |
| **AI Course Planner** | Generate a full 2-year quarter-by-quarter curriculum plan | Greedy optimization with prerequisite scheduling; Claude Haiku for plan explanation |

## Setup

### Prerequisites
- Python 3.10+
- An Anthropic API key (for AI Search and AI Course Planner features)

### Installation

```bash
cd "Deliverable 3"
pip install -r requirements.txt
```

### API Key Configuration

Create `.streamlit/secrets.toml` (if not already present):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

> The app works without an API key, but AI Search and AI Course Planner features will be disabled.

### Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Data Sources

| File | Description |
|------|-------------|
| `Data/Course Price History.xlsx` | Bid clearing prices across 11 quarters (Autumn 2023 – Spring 2026) |
| `Data/Booth_MBA_Course_Evaluation_Data.xlsx` | Student course evaluations (ratings, workload, clarity, etc.) |
| `Data/Course List.xlsx` | Current-year course catalog with schedules and instructors |
| `Data/Degree Requirements.rtf` | Foundation, FLMBE, and unit requirements for the MBA degree |
| `Data/Concentration Requirements.rtf` | Course requirements for each concentration |
| `Data/bid_model_v2.pkl` | Pre-trained two-stage prediction model (Random Forest) |

## Prediction Model

**Architecture**: Two-stage model trained on ~11 quarters of bid data.

- **Stage 1 — Demand Classifier**: Random Forest classifier predicts whether a course will have non-zero bidding demand (~96% accuracy).
- **Stage 2 — Price Regressor**: Random Forest / Gradient Boosting on log-transformed prices predicts clearing price for courses with demand.

Key features: historical price trends, instructor popularity, capacity/fill ratio, evaluation scores, schedule attributes.

## Built With

- [Streamlit](https://streamlit.io/) — UI framework
- [Claude Code](https://claude.ai/claude-code) — Primary development tool
- [Claude Haiku](https://docs.anthropic.com/en/docs/about-claude/models) — Natural language search & plan explanation
- [scikit-learn](https://scikit-learn.org/) — Bid prediction models
- [scipy](https://scipy.org/) — Bid allocation optimization (SLSQP)
- [Plotly](https://plotly.com/) — Interactive charts
