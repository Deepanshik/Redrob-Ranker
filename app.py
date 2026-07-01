import io
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import directly from rank.py — same functions, same logic
from rank import (
    JD_TEXT, WEIGHTS,
    detect_honeypot, score_skills, score_experience,
    score_availability, score_platform_trust, score_location,
    build_candidate_text, build_reasoning,
)

st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")

st.title("🎯 Redrob Candidate Ranking System")
st.caption("CPU-only · No API calls · Same logic as the submitted rank.py")

with st.expander("How it works"):
    st.markdown("""
    **Pipeline:**
    1. Upload a small `.jsonl` sample (or click the demo button)
    2. Honeypot filter removes fake profiles
    3. TF-IDF measures semantic similarity to the job description
    4. 5 scoring functions run: Skill · Experience · Availability · Platform · Location
    5. Weighted combination → ranked output

    **Weights:** Semantic 22% · Skill 28% · Experience 25% · Availability 15% · Platform 5% · Location 5%
    """)

# ── Sample data (50 varied candidates for demo) ───────────────────────
DEMO_CANDIDATES = [
    {
        "candidate_id": f"CAND_{str(i).zfill(7)}",
        "profile": {
            "anonymized_name": f"Candidate {i}",
            "headline": headline,
            "summary": summary,
            "current_title": title,
            "current_company": company,
            "current_industry": industry,
            "location": location,
            "country": "India",
            "years_of_experience": yoe,
        },
        "career_history": [{
            "company": company, "title": title,
            "start_date": "2020-01-01", "end_date": None,
            "duration_months": yoe * 10, "is_current": True,
            "industry": industry, "company_size": "201-500",
            "description": desc,
        }],
        "education": [],
        "skills": skills,
        "certifications": [],
        "languages": [],
        "redrob_signals": signals,
    }
    for i, (title, headline, summary, company, industry, location,
            yoe, desc, skills, signals) in enumerate([
        (
            "Senior Machine Learning Engineer",
            "Senior ML Engineer | Retrieval & Ranking | Vector Search",
            "7 years building production embedding-based retrieval and ranking systems.",
            "Redrob AI", "Technology", "Pune, Maharashtra", 7,
            "Built hybrid BM25+dense retrieval pipeline. Improved NDCG@10 by 18%. Deployed Pinecone vector index serving 2M users. A/B tested multiple ranking models.",
            [{"name":"Embeddings","proficiency":"expert","endorsements":45,"duration_months":48},
             {"name":"Python","proficiency":"expert","endorsements":60,"duration_months":84},
             {"name":"NDCG","proficiency":"advanced","endorsements":20,"duration_months":36},
             {"name":"Pinecone","proficiency":"advanced","endorsements":18,"duration_months":24},
             {"name":"Elasticsearch","proficiency":"advanced","endorsements":30,"duration_months":36}],
            {"profile_completeness_score":95,"signup_date":"2023-01-01","last_active_date":"2026-06-25",
             "open_to_work_flag":True,"profile_views_received_30d":80,"applications_submitted_30d":3,
             "recruiter_response_rate":0.85,"avg_response_time_hours":3,"skill_assessment_scores":{"Python":92,"Embeddings":89},
             "connection_count":350,"endorsements_received":90,"notice_period_days":30,
             "expected_salary_range_inr_lpa":{"min":35,"max":45},"preferred_work_mode":"hybrid",
             "willing_to_relocate":True,"github_activity_score":78,"search_appearance_30d":130,
             "saved_by_recruiters_30d":18,"interview_completion_rate":0.9,"offer_acceptance_rate":0.75,
             "verified_email":True,"verified_phone":True,"linkedin_connected":True},
        ),
        (
            "NLP Engineer",
            "NLP Engineer | Semantic Search | LLM Fine-tuning",
            "6 years in NLP, semantic search, and information retrieval systems.",
            "Startup X", "Technology", "Noida, Uttar Pradesh", 6,
            "Built semantic search system using sentence-transformers and FAISS. Fine-tuned BERT for ranking. Implemented BM25 hybrid retrieval.",
            [{"name":"NLP","proficiency":"expert","endorsements":38,"duration_months":60},
             {"name":"FAISS","proficiency":"advanced","endorsements":22,"duration_months":30},
             {"name":"Python","proficiency":"expert","endorsements":55,"duration_months":72},
             {"name":"Sentence-transformers","proficiency":"advanced","endorsements":16,"duration_months":24},
             {"name":"Retrieval","proficiency":"advanced","endorsements":20,"duration_months":36}],
            {"profile_completeness_score":90,"signup_date":"2023-03-01","last_active_date":"2026-06-20",
             "open_to_work_flag":True,"profile_views_received_30d":60,"applications_submitted_30d":4,
             "recruiter_response_rate":0.75,"avg_response_time_hours":6,"skill_assessment_scores":{"Python":88,"NLP":85},
             "connection_count":280,"endorsements_received":75,"notice_period_days":15,
             "expected_salary_range_inr_lpa":{"min":28,"max":38},"preferred_work_mode":"hybrid",
             "willing_to_relocate":False,"github_activity_score":65,"search_appearance_30d":90,
             "saved_by_recruiters_30d":12,"interview_completion_rate":0.85,"offer_acceptance_rate":0.8,
             "verified_email":True,"verified_phone":True,"linkedin_connected":True},
        ),
        (
            "Marketing Manager",
            "Marketing Manager | Growth | Brand Strategy",
            "8 years in digital marketing and brand management.",
            "Brand Co", "Marketing", "Delhi, Delhi", 8,
            "Led marketing campaigns and brand strategy. Managed social media presence.",
            [{"name":"Marketing","proficiency":"expert","endorsements":40,"duration_months":96},
             {"name":"Python","proficiency":"beginner","endorsements":2,"duration_months":3},
             {"name":"Embeddings","proficiency":"beginner","endorsements":0,"duration_months":0}],
            {"profile_completeness_score":70,"signup_date":"2022-06-01","last_active_date":"2025-10-01",
             "open_to_work_flag":False,"profile_views_received_30d":10,"applications_submitted_30d":1,
             "recruiter_response_rate":0.2,"avg_response_time_hours":72,"skill_assessment_scores":{},
             "connection_count":150,"endorsements_received":30,"notice_period_days":90,
             "expected_salary_range_inr_lpa":{"min":15,"max":22},"preferred_work_mode":"onsite",
             "willing_to_relocate":False,"github_activity_score":-1,"search_appearance_30d":20,
             "saved_by_recruiters_30d":1,"interview_completion_rate":0.5,"offer_acceptance_rate":0.3,
             "verified_email":True,"verified_phone":False,"linkedin_connected":True},
        ),
        (
            "Data Scientist",
            "Data Scientist | ML | Recommendation Systems",
            "5 years building recommendation and ranking systems at product companies.",
            "Product Co", "Technology", "Bangalore, Karnataka", 5,
            "Built item-item collaborative filtering recommendation engine. Deployed ranking models for search. Used XGBoost for learning-to-rank.",
            [{"name":"Python","proficiency":"expert","endorsements":48,"duration_months":60},
             {"name":"Ranking","proficiency":"advanced","endorsements":25,"duration_months":36},
             {"name":"XGBoost","proficiency":"advanced","endorsements":20,"duration_months":30},
             {"name":"Recommendation","proficiency":"advanced","endorsements":22,"duration_months":36},
             {"name":"Elasticsearch","proficiency":"intermediate","endorsements":12,"duration_months":18}],
            {"profile_completeness_score":88,"signup_date":"2023-05-01","last_active_date":"2026-06-15",
             "open_to_work_flag":True,"profile_views_received_30d":55,"applications_submitted_30d":2,
             "recruiter_response_rate":0.7,"avg_response_time_hours":8,"skill_assessment_scores":{"Python":85},
             "connection_count":260,"endorsements_received":68,"notice_period_days":30,
             "expected_salary_range_inr_lpa":{"min":25,"max":35},"preferred_work_mode":"hybrid",
             "willing_to_relocate":True,"github_activity_score":55,"search_appearance_30d":75,
             "saved_by_recruiters_30d":10,"interview_completion_rate":0.8,"offer_acceptance_rate":0.7,
             "verified_email":True,"verified_phone":True,"linkedin_connected":True},
        ),
        (
            "Operations Manager",
            "Operations Manager | Supply Chain | Process Optimization",
            "9 years managing operations and supply chain processes.",
            "Ops Corp", "Operations", "Mumbai, Maharashtra", 9,
            "Managed team of 30 in logistics and supply chain. Reduced costs by 15%.",
            [{"name":"Operations","proficiency":"expert","endorsements":35,"duration_months":108},
             {"name":"Python","proficiency":"beginner","endorsements":1,"duration_months":2},
             {"name":"Vector Database","proficiency":"beginner","endorsements":0,"duration_months":0}],
            {"profile_completeness_score":65,"signup_date":"2022-01-01","last_active_date":"2025-08-01",
             "open_to_work_flag":False,"profile_views_received_30d":8,"applications_submitted_30d":0,
             "recruiter_response_rate":0.1,"avg_response_time_hours":120,"skill_assessment_scores":{},
             "connection_count":100,"endorsements_received":20,"notice_period_days":90,
             "expected_salary_range_inr_lpa":{"min":18,"max":25},"preferred_work_mode":"onsite",
             "willing_to_relocate":False,"github_activity_score":-1,"search_appearance_30d":5,
             "saved_by_recruiters_30d":0,"interview_completion_rate":0.4,"offer_acceptance_rate":0.2,
             "verified_email":True,"verified_phone":False,"linkedin_connected":False},
        ),
    ], 1)
]


def run_ranking_on_candidates(candidates):
    """Core ranking logic — mirrors rank.py main() exactly."""

    # Honeypot filter
    filtered = [c for c in candidates if not detect_honeypot(c)]

    if not filtered:
        return []

    # TF-IDF semantic scoring
    texts = [build_candidate_text(c) for c in filtered]
    all_texts = [JD_TEXT] + texts
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                                  sublinear_tf=True, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = tfidf_matrix[0]
    candidate_vecs = tfidf_matrix[1:]
    sem_scores = cosine_similarity(candidate_vecs, jd_vec).flatten()

    # Full scoring
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
            "comp":         comp,
            "candidate":    candidate,
        })

    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    return results


# ── Sidebar: input source ─────────────────────────────────────────────
st.sidebar.header("Input")
input_mode = st.sidebar.radio(
    "Candidate source",
    ["Use built-in demo (5 candidates)", "Upload your own .jsonl file"],
)

candidates = []

if input_mode == "Use built-in demo (5 candidates)":
    candidates = DEMO_CANDIDATES
    st.sidebar.success(f"{len(candidates)} demo candidates loaded.")

else:
    uploaded = st.sidebar.file_uploader(
        "Upload candidates.jsonl",
        type=["jsonl", "json"],
        help="One JSON object per line. Keep under 200MB for this demo.",
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
        st.sidebar.success(f"Loaded {len(candidates)} candidates.")
    else:
        st.sidebar.info("Upload a .jsonl file to begin, or switch to the demo.")

# ── Main: run and display ─────────────────────────────────────────────
if candidates:
    top_n = st.slider("Number of top candidates to display", 1,
                       min(50, len(candidates)), min(10, len(candidates)))

    if st.button("🚀 Run Ranker", type="primary", use_container_width=True):
        with st.spinner("Ranking candidates..."):
            t0 = time.time()
            results = run_ranking_on_candidates(candidates)
            elapsed = time.time() - t0

        honeypots = len(candidates) - len(results)

        st.success(
            f"Ranked {len(results)} candidates in {elapsed:.2f}s"
            + (f" · {honeypots} honeypot(s) filtered out" if honeypots else "")
        )

        if not results:
            st.warning("No candidates passed the honeypot filter. "
                       "Try uploading different data.")
            st.stop()

        top_results = results[:top_n]

        # ── Results table ──────────────────────────────────────────
        st.subheader("Ranked candidates")
        table_rows = []
        for rank, r in enumerate(top_results, 1):
            p = r["candidate"]["profile"]
            table_rows.append({
                "Rank":         rank,
                "Candidate ID": r["candidate_id"],
                "Title":        p.get("current_title", ""),
                "YoE":          int(p.get("years_of_experience", 0)),
                "Location":     p.get("location", ""),
                "Final Score":  r["final_score"],
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True,
                     hide_index=True)

        # ── Component breakdown ────────────────────────────────────
        st.subheader("Score breakdown — top candidate")
        top = top_results[0]
        comp_data = {
            "Semantic":     top["comp"]["semantic"],
            "Skill":        top["comp"]["skill"],
            "Experience":   top["comp"]["experience"],
            "Availability": top["comp"]["availability"],
            "Platform":     top["comp"]["platform"],
            "Location":     top["comp"]["location"],
        }
        comp_df = pd.DataFrame.from_dict(
            comp_data, orient="index", columns=["Score (0-100)"]
        )
        st.bar_chart(comp_df)
        st.markdown(
            f"**Reasoning:** "
            f"{build_reasoning(top['candidate'], top['comp'], top['final_score'])}"
        )

        # ── Reasoning for all top results ──────────────────────────
        with st.expander(f"Reasoning for all top {len(top_results)} candidates"):
            for rank, r in enumerate(top_results, 1):
                p = r["candidate"]["profile"]
                st.markdown(
                    f"**#{rank} {r['candidate_id']} — "
                    f"{p.get('current_title', '')}** "
                    f"(score: {r['final_score']})"
                )
                st.write(build_reasoning(r["candidate"], r["comp"], r["final_score"]))
                st.divider()

        # ── Download CSV ───────────────────────────────────────────
        st.subheader("Download submission CSV")
        top100 = results[:100]
        max_s = top100[0]["final_score"]
        min_s = top100[-1]["final_score"]
        rng   = max_s - min_s if max_s > min_s else 1.0

        csv_rows = []
        prev_norm = None
        for rank, r in enumerate(top100, 1):
            norm = round(0.40 + ((r["final_score"] - min_s) / rng) * 0.59, 4)
            if prev_norm is not None and norm > prev_norm:
                norm = prev_norm
            prev_norm = norm
            reasoning = build_reasoning(r["candidate"], r["comp"], r["final_score"])
            csv_rows.append([r["candidate_id"], rank, norm, reasoning])

        csv_df = pd.DataFrame(csv_rows,
                               columns=["candidate_id", "rank", "score", "reasoning"])
        buf = io.StringIO()
        csv_df.to_csv(buf, index=False)

        st.download_button(
            "⬇️ Download submission.csv (top 100)",
            data=buf.getvalue(),
            file_name="submission.csv",
            mime="text/csv",
        )

else:
    st.info("Select a candidate source from the left sidebar to get started.")

st.divider()
st.caption(
    "Redrob Intelligent Candidate Discovery & Ranking Challenge · "
    "CPU-only · No API calls · Reproducible with: "
    "`python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv`"
)