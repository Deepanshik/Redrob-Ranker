# Redrob Intelligent Candidate Ranking System

**Challenge:** Intelligent Candidate Discovery & Ranking Challenge  
**Approach:** Multi-dimensional rule-based + TF-IDF semantic scoring, CPU-only, no API calls

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** `scikit-learn`, `numpy` — no GPU, no LLM API keys needed.

---

## Reproduce Submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**Runtime:** ~12 seconds on CPU for 100K candidates (well within the 5-minute limit).

---

## How It Works

### Architecture

```
Job Description (text)
        │
        ▼
[TF-IDF Vectorizer]  (fit on JD + candidate texts)
        │
        ├──► Semantic Similarity Score (cosine sim vs JD)
        │
[Candidate Pool: 100K]
        │
        ▼
[Quick Pre-filter]  (rule-based, narrows to top 5K)
  - Negative title signals eliminated
  - No honeypots
  - YoE and skill count signal
        │
        ▼
[5-Component Scorer]  (for each of 5K candidates)
  A. Semantic Fit     25%  TF-IDF cosine similarity to JD
  B. Skill Quality    28%  proficiency + endorsements + duration + assessments
  C. Experience       22%  YoE + title fit + career trajectory + stability
  D. Availability     15%  open_to_work + recency + notice + response rate
  E. Platform Trust    5%  completeness + verified + github + social proof
  F. Location          5%  Pune/Noida/target city preference
        │
        ▼
[Weighted Score → Sort → Top 100]
        │
        ▼
[submission.csv]
```

### Scoring Details

#### A. Semantic Fit (25%)
TF-IDF bigram vectorizer over candidate's headline, summary, title, skills, and career descriptions. Cosine similarity against a dense JD text that emphasizes key terms. No external model download required.

#### B. Skill Quality (28%)
- Required skills (embeddings, vector DB, retrieval, ranking, NLP, Python, NDCG…): up to 45 pts
- Preferred skills (LoRA, LTR, RAG, PyTorch…): up to 15 pts
- Proficiency level of matched skills: up to 20 pts
- Endorsements (social proof): up to 10 pts
- Duration months (genuine usage vs listing): up to 5 pts
- Platform assessment scores: up to 5 pts

#### C. Experience (22%)
- **YoE:** 5–9 year sweet spot scores maximum; penalties for <3 or >10
- **Title:** AI/ML/data/search engineer titles score high; marketing, HR, ops titles penalized
- **Career trajectory:** product company roles rewarded; all-consulting career penalized (per JD)
- **Stability:** avg tenure ≥24 months rewarded (JD wants 3+ year commitment)

#### D. Availability (15%)
- `open_to_work_flag`: 20 pts
- Last active recency: 25 pts (inactive 180+ days → 0)
- Notice period: ≤30 days scores well; 90+ days → 0 (per JD)
- Recruiter response rate + response time: 20 pts
- Interview completion + offer acceptance: 15 pts

#### E. Platform Trust (5%)
Profile completeness, email/phone verification, LinkedIn, GitHub activity score, connections, endorsements received, recruiter demand (saved/searches).

#### F. Location (5%)
Pune/Noida → 100; Hyderabad/Mumbai/Delhi/Bangalore → 80; other India + willing to relocate → 65.

### Honeypot Detection
Candidates are flagged and excluded if:
1. 5+ "expert" skills with 0 months duration (impossible)
2. Stated YoE > 3× actual career history in months

### JD Interpretation
The JD explicitly warns:
- Keyword matching is a trap — the system must understand what profiles *mean*
- Inactive candidates (low response rate, not open-to-work) are down-weighted
- All-consulting careers are penalized
- Title/role mismatches are hard disqualifiers regardless of skills listed

---

## File Structure

```
redrob-ranker/
├── rank.py                       # Main entry point
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── submission_metadata.yaml      # Team metadata
├── output/
│   └── submission.csv            # Final ranked output
└── validate_submission.py        # Format validator
```

---

## Validate Output

```bash
python validate_submission.py output/submission.csv
```

---

## Design Decisions

**Why TF-IDF instead of sentence-transformers?**  
The compute constraints require no network access during ranking and ≤5 min on CPU. TF-IDF with bigrams runs in ~2 seconds on 5K candidates and requires no model download, making it fully reproducible in any sandboxed environment. For production, this can be swapped with a cached embedding index (offline pre-computation allowed).

**Why pre-filter to 5K?**  
100K full candidates × full scoring = ~60 seconds still within budget, but pre-filtering to 5K reduces TF-IDF matrix size 20× and keeps runtime to ~12 seconds, leaving headroom for larger datasets.

**Why these weights?**  
Skills (28%) and experience (22%) are the core fit signals. Semantic (25%) captures implicit matches the rule-based system misses. Availability (15%) implements the JD's explicit instruction to down-weight passive candidates. Location and platform trust are tie-breakers.

## 📄 Presentation Deck
[View Technical Architecture PDF](docs/Redrob_Ranking_Deck_Final.pdf)