# Redrob Intelligent Candidate Ranking System

**Challenge:** Intelligent Candidate Discovery & Ranking Challenge — Redrob AI × India Runs  
**Team:** Deepanshi Khandelwal  
**Approach:** Multi-dimensional rule-based + TF-IDF semantic scoring, CPU-only, no API calls

---

## 📎 Submission Assets

| Asset | Link |
|---|---|
| 🐙 GitHub Repo | [github.com/Deepanshik/Redrob-Ranker](https://github.com/Deepanshik/Redrob-Ranker) |
| 🚀 Live Sandbox | [redrob-ranker.streamlit.app](https://redrob-ranker-ghdyvc8ojcsm28syn99gse.streamlit.app/) |
| 📊 Ranked Output | `output/submission.xlsx` — 100 rows, ranks 1–100 |
| 📄 Presentation Deck | [docs/Redrob-Ranking-Deck-Final.pdf](docs/Redrob-Ranking-Deck-Final.pdf) |

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
python validate_submission.py ./output/submission.csv
```

**Runtime:** 19 seconds on CPU for 100K candidates. No GPU. No API calls.

---

## 🏆 Top 5 Results

| Rank | Candidate ID | Role | Company | Location | Score |
|---|---|---|---|---|---|
| #1 | CAND_0018499 | Sr. ML Engineer | Zomato | Noida, UP | 0.9900 |
| #2 | CAND_0086022 | Sr. Applied Scientist | Sarvam AI | Kolkata, WB | 0.8664 |
| #3 | CAND_0088025 | Staff ML Engineer | Yellow.ai | Jaipur, RJ | 0.8631 |
| #4 | CAND_0027691 | NLP Engineer | Haptik | Pune, MH | 0.7748 |
| #5 | CAND_0081846 | Lead AI Engineer | Razorpay | Jaipur, RJ | 0.7620 |

---

## 🏗️ Architecture

```
candidates.jsonl (100K)
        │
        ▼
[Honeypot Filter]       detect_honeypot() — removes ~80 fake profiles
        │
        ▼
[Quick Pre-filter]      rule-based quick_score() — 100K → 5K
        │
        ▼
[TF-IDF Semantic]       bigram vectorizer — cosine similarity vs JD_TEXT
        │
        ▼
[5-Dimension Scorer]
  Skill Quality    28%  proficiency × duration × endorsements
  Experience       25%  trajectory + title + YoE + stability
  Semantic TF-IDF  22%  implicit JD match
  Availability     15%  open_to_work + last active + notice + response rate
  Platform Trust    5%  completeness + github + verification
  Location          5%  Pune/Noida preferred
        │
        ▼
[Weighted Sum → Sort → Top 100 → submission.xlsx]
```

**Final Score Formula:**
```
Skill×0.28 + Exp×0.25 + Sem×0.22 + Avail×0.15 + Plat×0.05 + Loc×0.05
```

---

## 🔍 Key Design Decisions

**Why TF-IDF instead of sentence-transformers?**  
Compute constraints require no network access during ranking and ≤5 min on CPU. TF-IDF bigrams run in under 2 seconds on 5K candidates with no model download — fully reproducible in any sandboxed Docker container.

**Why pre-filter to 5K?**  
Avoids running expensive scoring on obviously wrong candidates (marketing managers, civil engineers). Cheap rule-based scoring narrows 100K → 5K in seconds, then careful multi-dimensional scoring runs on the realistic candidate pool.

**Why these weights?**  
Grounded directly in JD language: skills (28%) and experience (25%) are non-negotiable hard filters. Availability (15%) is explicitly called out in the JD — *"a candidate with 5% recruiter response rate is not actually available."* Location and platform trust are tiebreakers.

**Honeypot detection:**  
Two consistency checks before any scoring:
1. 5+ "expert" skills with 0 months usage → impossible claim → dropped
2. Stated YoE > 3.5× actual career history → inconsistent data → dropped

---

## 📁 File Structure

```
RedrobRanker/
├── rank.py                    # Main ranker — all scoring logic
├── app.py                     # Streamlit sandbox demo
├── requirements.txt           # scikit-learn, numpy, pandas, streamlit
├── validate_submission.py     # Official format validator
├── submission-metada.yaml     # Team metadata + sandbox link
├── README.md                  # This file
├── data/
│   └── candidates.jsonl       # 100K candidate profiles (local only)
├── docs/
│   └── Redrob-Ranking-Deck-Final.pdf   # Presentation deck
└── output/
    ├── submission.csv         # Validated ranked output (CSV)
    └── submission.xlsx        # Ranked output with formatting (XLSX)
```

---

## ✅ Validation

```bash
python validate_submission.py ./output/submission.csv
# ✅ Submission is VALID.
```

Passes all checks: 100 rows, valid CAND_XXXXXXX IDs, ranks 1–100 each exactly once, scores non-increasing, tie-break by candidate_id ascending.