import io
import json
import time

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank import (
    JD_TEXT, WEIGHTS,
    detect_honeypot, score_skills, score_experience,
    score_availability, score_platform_trust, score_location,
    build_candidate_text, build_reasoning,
)

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")
st.title("🎯 Redrob Candidate Ranking System")
st.caption("CPU-only · No API calls · Same scoring logic as submitted rank.py")

with st.expander("How it works"):
    st.markdown("""
    **Pipeline:**
    1. Candidates loaded (upload a small sample or use built-in demo)
    2. Honeypot filter removes fake/impossible profiles
    3. TF-IDF bigrams measure semantic similarity to the job description
    4. 5 scoring dimensions: Skill · Experience · Availability · Platform · Location
    5. Weighted combination → ranked output

    **Weights:** Semantic 22% · Skill 28% · Experience 25% · Availability 15% · Platform+Location 10%
    """)

# ── Built-in demo candidates ──────────────────────────────────────────
DEMO = [
    {
        "candidate_id": "CAND_0000001",
        "profile": {"anonymized_name": "Demo A", "headline": "Senior ML Engineer | Retrieval & Ranking | Vector Search",
                    "summary": "7 years building production embedding retrieval ranking systems at product companies.",
                    "current_title": "Senior Machine Learning Engineer", "current_company": "StartupX",
                    "current_industry": "Technology", "location": "Pune, Maharashtra",
                    "country": "India", "years_of_experience": 7},
        "career_history": [{"company": "StartupX", "title": "Senior Machine Learning Engineer",
                             "start_date": "2020-01-01", "end_date": None, "duration_months": 42,
                             "is_current": True, "industry": "Technology", "company_size": "201-500",
                             "description": "Built hybrid BM25 dense retrieval pipeline. Deployed Pinecone vector index serving 2M users. Improved NDCG@10 by 18% through A/B testing reranking models. Production embedding drift monitoring."}],
        "education": [], "certifications": [], "languages": [],
        "skills": [{"name": "Embeddings", "proficiency": "expert", "endorsements": 45, "duration_months": 48},
                   {"name": "Python", "proficiency": "expert", "endorsements": 60, "duration_months": 84},
                   {"name": "NDCG", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
                   {"name": "Pinecone", "proficiency": "advanced", "endorsements": 18, "duration_months": 24},
                   {"name": "Elasticsearch", "proficiency": "advanced", "endorsements": 30, "duration_months": 36},
                   {"name": "Retrieval", "proficiency": "expert", "endorsements": 35, "duration_months": 42}],
        "redrob_signals": {"profile_completeness_score": 95, "signup_date": "2023-01-01",
                           "last_active_date": "2026-06-25", "open_to_work_flag": True,
                           "profile_views_received_30d": 80, "applications_submitted_30d": 3,
                           "recruiter_response_rate": 0.85, "avg_response_time_hours": 3,
                           "skill_assessment_scores": {"Python": 92, "Embeddings": 89},
                           "connection_count": 350, "endorsements_received": 90,
                           "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 35, "max": 45},
                           "preferred_work_mode": "hybrid", "willing_to_relocate": True,
                           "github_activity_score": 78, "search_appearance_30d": 130,
                           "saved_by_recruiters_30d": 18, "interview_completion_rate": 0.9,
                           "offer_acceptance_rate": 0.75, "verified_email": True,
                           "verified_phone": True, "linkedin_connected": True},
    },
    {
        "candidate_id": "CAND_0000002",
        "profile": {"anonymized_name": "Demo B", "headline": "NLP Engineer | Semantic Search | FAISS | BM25",
                    "summary": "6 years in NLP, semantic search, and information retrieval at product companies.",
                    "current_title": "NLP Engineer", "current_company": "TechCo",
                    "current_industry": "Technology", "location": "Noida, Uttar Pradesh",
                    "country": "India", "years_of_experience": 6},
        "career_history": [{"company": "TechCo", "title": "NLP Engineer",
                             "start_date": "2021-01-01", "end_date": None, "duration_months": 36,
                             "is_current": True, "industry": "Technology", "company_size": "501-1000",
                             "description": "Built semantic search using sentence-transformers and FAISS. Fine-tuned BERT for ranking. Implemented BM25 hybrid retrieval. Shipped to production serving real users."}],
        "education": [], "certifications": [], "languages": [],
        "skills": [{"name": "NLP", "proficiency": "expert", "endorsements": 38, "duration_months": 60},
                   {"name": "FAISS", "proficiency": "advanced", "endorsements": 22, "duration_months": 30},
                   {"name": "Python", "proficiency": "expert", "endorsements": 55, "duration_months": 72},
                   {"name": "Sentence-transformers", "proficiency": "advanced", "endorsements": 16, "duration_months": 24},
                   {"name": "BM25", "proficiency": "advanced", "endorsements": 14, "duration_months": 24}],
        "redrob_signals": {"profile_completeness_score": 90, "signup_date": "2023-03-01",
                           "last_active_date": "2026-06-20", "open_to_work_flag": True,
                           "profile_views_received_30d": 60, "applications_submitted_30d": 4,
                           "recruiter_response_rate": 0.75, "avg_response_time_hours": 6,
                           "skill_assessment_scores": {"Python": 88, "NLP": 85},
                           "connection_count": 280, "endorsements_received": 75,
                           "notice_period_days": 15, "expected_salary_range_inr_lpa": {"min": 28, "max": 38},
                           "preferred_work_mode": "hybrid", "willing_to_relocate": False,
                           "github_activity_score": 65, "search_appearance_30d": 90,
                           "saved_by_recruiters_30d": 12, "interview_completion_rate": 0.85,
                           "offer_acceptance_rate": 0.8, "verified_email": True,
                           "verified_phone": True, "linkedin_connected": True},
    },
    {
        "candidate_id": "CAND_0000003",
        "profile": {"anonymized_name": "Demo C", "headline": "Marketing Manager | Brand Strategy | Growth",
                    "summary": "8 years in digital marketing and brand management.",
                    "current_title": "Marketing Manager", "current_company": "BrandCo",
                    "current_industry": "Marketing", "location": "Delhi, Delhi",
                    "country": "India", "years_of_experience": 8},
        "career_history": [{"company": "BrandCo", "title": "Marketing Manager",
                             "start_date": "2018-01-01", "end_date": None, "duration_months": 72,
                             "is_current": True, "industry": "Marketing", "company_size": "201-500",
                             "description": "Led marketing campaigns. Managed social media and brand strategy."}],
        "education": [], "certifications": [], "languages": [],
        "skills": [{"name": "Marketing", "proficiency": "expert", "endorsements": 40, "duration_months": 96},
                   {"name": "Python", "proficiency": "beginner", "endorsements": 2, "duration_months": 3},
                   {"name": "Embeddings", "proficiency": "beginner", "endorsements": 0, "duration_months": 0}],
        "redrob_signals": {"profile_completeness_score": 70, "signup_date": "2022-06-01",
                           "last_active_date": "2025-10-01", "open_to_work_flag": False,
                           "profile_views_received_30d": 10, "applications_submitted_30d": 1,
                           "recruiter_response_rate": 0.2, "avg_response_time_hours": 72,
                           "skill_assessment_scores": {}, "connection_count": 150,
                           "endorsements_received": 30, "notice_period_days": 90,
                           "expected_salary_range_inr_lpa": {"min": 15, "max": 22},
                           "preferred_work_mode": "onsite", "willing_to_relocate": False,
                           "github_activity_score": -1, "search_appearance_30d": 20,
                           "saved_by_recruiters_30d": 1, "interview_completion_rate": 0.5,
                           "offer_acceptance_rate": 0.3, "verified_email": True,
                           "verified_phone": False, "linkedin_connected": True},
    },
    {
        "candidate_id": "CAND_0000004",
        "profile": {"anonymized_name": "Demo D", "headline": "Data Scientist | Recommendation Systems | LTR",
                    "summary": "5 years building recommendation and ranking systems at product companies.",
                    "current_title": "Data Scientist", "current_company": "ProductCo",
                    "current_industry": "Technology", "location": "Bangalore, Karnataka",
                    "country": "India", "years_of_experience": 5},
        "career_history": [{"company": "ProductCo", "title": "Data Scientist",
                             "start_date": "2021-01-01", "end_date": None, "duration_months": 36,
                             "is_current": True, "industry": "Technology", "company_size": "1001-5000",
                             "description": "Built item-item collaborative filtering recommendation engine. Deployed XGBoost learning-to-rank model for search. A/B tested ranking algorithms in production."}],
        "education": [], "certifications": [], "languages": [],
        "skills": [{"name": "Python", "proficiency": "expert", "endorsements": 48, "duration_months": 60},
                   {"name": "Ranking", "proficiency": "advanced", "endorsements": 25, "duration_months": 36},
                   {"name": "XGBoost", "proficiency": "advanced", "endorsements": 20, "duration_months": 30},
                   {"name": "Elasticsearch", "proficiency": "intermediate", "endorsements": 12, "duration_months": 18}],
        "redrob_signals": {"profile_completeness_score": 88, "signup_date": "2023-05-01",
                           "last_active_date": "2026-06-15", "open_to_work_flag": True,
                           "profile_views_received_30d": 55, "applications_submitted_30d": 2,
                           "recruiter_response_rate": 0.7, "avg_response_time_hours": 8,
                           "skill_assessment_scores": {"Python": 85},
                           "connection_count": 260, "endorsements_received": 68,
                           "notice_period_days": 30, "expected_salary_range_inr_lpa": {"min": 25, "max": 35},
                           "preferred_work_mode": "hybrid", "willing_to_relocate": True,
                           "github_activity_score": 55, "search_appearance_30d": 75,
                           "saved_by_recruiters_30d": 10, "interview_completion_rate": 0.8,
                           "offer_acceptance_rate": 0.7, "verified_email": True,
                           "verified_phone": True, "linkedin_connected": True},
    },
    {
        "candidate_id": "CAND_0000005",
        "profile": {"anonymized_name": "Demo E", "headline": "Operations Manager | Supply Chain | Process Optimization",
                    "summary": "9 years managing operations and supply chain.",
                    "current_title": "Operations Manager", "current_company": "OpsCorp",
                    "current_industry": "Operations", "location": "Mumbai, Maharashtra",
                    "country": "India", "years_of_experience": 9},
        "career_history": [{"company": "OpsCorp", "title": "Operations Manager",
                             "start_date": "2015-01-01", "end_date": None, "duration_months": 108,
                             "is_current": True, "industry": "Operations", "company_size": "1001-5000",
                             "description": "Managed team of 30 in logistics and supply chain. Reduced costs by 15%."}],
        "education": [], "certifications": [], "languages": [],
        "skills": [{"name": "Operations", "proficiency": "expert", "endorsements": 35, "duration_months": 108},
                   {"name": "Python", "proficiency": "beginner", "endorsements": 1, "duration_months": 2}],
        "redrob_signals": {"profile_completeness_score": 65, "signup_date": "2022-01-01",
                           "last_active_date": "2025-08-01", "open_to_work_flag": False,
                           "profile_views_received_30d": 8, "applications_submitted_30d": 0,
                           "recruiter_response_rate": 0.1, "avg_response_time_hours": 120,
                           "skill_assessment_scores": {}, "connection_count": 100,
                           "endorsements_received": 20, "notice_period_days": 90,
                           "expected_salary_range_inr_lpa": {"min": 18, "max": 25},
                           "preferred_work_mode": "onsite", "willing_to_relocate": False,
                           "github_activity_score": -1, "search_appearance_30d": 5,
                           "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.4,
                           "offer_acceptance_rate": 0.2, "verified_email": True,
                           "verified_phone": False, "linkedin_connected": False},
    },
]


def rank_candidates(candidates):
    """Rank a list of candidate dicts. Same logic as run_ranking() in rank.py."""
    filtered = [c for c in candidates if not detect_honeypot(c)]
    if not filtered:
        return []

    texts = [build_candidate_text(c) for c in filtered]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                  sublinear_tf=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform([JD_TEXT] + texts)
    sem_scores = cosine_similarity(tfidf_matrix[1:], tfidf_matrix[0]).flatten()

    results = []
    for candidate, sem in zip(filtered, sem_scores):
        comp = {
            "semantic":      float(sem) * 100,
            "skill":         score_skills(candidate),
            "experience":    score_experience(candidate),
            "availability":  score_availability(candidate),
            "platform":      score_platform_trust(candidate),
            "location":      score_location(candidate),
        }
        final = sum(comp[k] * v for k, v in WEIGHTS.items())
        results.append({
            "candidate_id": candidate.get("candidate_id", ""),
            "final_score":  round(max(0.0, min(100.0, final)), 2),
            "comp": comp,
            "candidate": candidate,
        })

    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    return results


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Input")
mode = st.sidebar.radio("Candidate source",
                         ["Built-in demo (5 candidates)",
                          "Upload your own .jsonl file"])

candidates = []

if mode == "Built-in demo (5 candidates)":
    candidates = DEMO
    st.sidebar.success("5 demo candidates loaded.")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a small candidates.jsonl sample",
        type=["jsonl", "json"],
        help="Keep under 10MB for smooth demo. One JSON object per line."
    )
    if uploaded:
        raw = uploaded.read().decode("utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if candidates:
            st.sidebar.success(f"Loaded {len(candidates)} candidates.")
        else:
            st.sidebar.error("Could not parse any candidates from the file.")
    else:
        st.sidebar.info("Upload a .jsonl file or switch to demo.")

# ── Main area ─────────────────────────────────────────────────────────
if candidates:
    top_n = st.slider("Top N candidates to show",
                       1, min(50, len(candidates)),
                       min(10, len(candidates)))

    if st.button("🚀 Run Ranker", type="primary", use_container_width=True):
        with st.spinner("Ranking candidates..."):
            t0 = time.time()
            results = rank_candidates(candidates)
            elapsed = time.time() - t0

        honeypots = len(candidates) - len(results)
        st.success(
            f"Ranked {len(results)} candidates in {elapsed:.2f}s"
            + (f" · {honeypots} honeypot(s) removed" if honeypots else "")
        )

        if not results:
            st.warning("No valid candidates after filtering.")
            st.stop()

        top = results[:top_n]

        # Results table
        st.subheader("Ranked Candidates")
        st.dataframe(pd.DataFrame([{
            "Rank": i + 1,
            "Candidate ID": r["candidate_id"],
            "Title": r["candidate"]["profile"].get("current_title", ""),
            "YoE": int(r["candidate"]["profile"].get("years_of_experience", 0)),
            "Location": r["candidate"]["profile"].get("location", ""),
            "Score": r["final_score"],
        } for i, r in enumerate(top)]), use_container_width=True, hide_index=True)

        # Score breakdown chart
        st.subheader(f"Score Breakdown — #{1} {top[0]['candidate_id']}")
        st.bar_chart(pd.DataFrame.from_dict(
            {k: [v] for k, v in top[0]["comp"].items()},
            orient="columns"
        ))
        st.markdown(f"**Reasoning:** {build_reasoning(top[0]['candidate'], top[0]['comp'], top[0]['final_score'])}")

        # Reasoning expander
        with st.expander("Reasoning for all ranked candidates"):
            for i, r in enumerate(top):
                p = r["candidate"]["profile"]
                st.markdown(f"**#{i+1} {r['candidate_id']} — {p.get('current_title','')}** (score: {r['final_score']})")
                st.write(build_reasoning(r["candidate"], r["comp"], r["final_score"]))
                st.divider()

        # Download CSV
        st.subheader("Download Submission CSV")
        top100 = results[:100]
        max_s = top100[0]["final_score"]
        min_s = top100[-1]["final_score"]
        rng = max_s - min_s if max_s > min_s else 1.0
        rows, prev = [], None
        for rank, r in enumerate(top100, 1):
            norm = round(0.40 + ((r["final_score"] - min_s) / rng) * 0.59, 4)
            if prev and norm > prev:
                norm = prev
            prev = norm
            rows.append([r["candidate_id"], rank, norm,
                         build_reasoning(r["candidate"], r["comp"], r["final_score"])])

        buf = io.StringIO()
        pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"]).to_csv(buf, index=False)
        st.download_button("⬇️ Download submission.csv", data=buf.getvalue(),
                           file_name="submission.csv", mime="text/csv")
else:
    st.info("👈 Select a candidate source from the left sidebar to get started.")

st.divider()
st.caption("Redrob Candidate Ranking Challenge · CPU-only · No API calls · "
           "Reproduce: python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv")