#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranking System
Job: Senior AI Engineer — Founding Team at Redrob AI (Series A)

KEY INSIGHTS FROM ACTUAL JD (read carefully before modifying weights):

1. "The right answer involves reasoning about the gap between what the JD says
   and what the JD means." — explicit hackathon hint
2. Keyword matching is THE TRAP. A marketing manager with AI skills ≠ fit.
3. Behavioral signals are first-class: inactive 6mo + 5% response rate = not available.
4. Production experience >> academic/research experience
5. Product company experience >> consulting company
6. "We'd rather see 10 great matches than 1000 maybes." — precision over recall

EXPLICIT DISQUALIFIERS from JD:
 - Pure research roles, no production deployment
 - LangChain-only recent (<12mo) AI experience
 - Title chasers (avg tenure < 18mo across companies)
 - Consulting-only career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini etc.)
 - Not written code in 18 months (pure architecture/lead roles)
 - CV/Speech/Robotics without NLP/IR
 - Closed-source only for 5+ years

Compute constraints: CPU only, no network, no API calls, < 5 min on 16GB RAM.
"""

import argparse
import csv
import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TODAY = date.today()

# ─────────────────────── JD TEXT (tuned to actual JD) ─────────────────── #

JD_TEXT = (
    "Senior AI Engineer founding team Redrob AI Series A talent intelligence platform "
    "Pune Noida India hybrid 5 9 years applied machine learning production systems "
    "embeddings retrieval ranking matching search NLP information retrieval "
    "sentence-transformers OpenAI embeddings BGE E5 embedding drift index refresh "
    "retrieval quality regression production vector database Pinecone Weaviate Qdrant "
    "Milvus FAISS OpenSearch Elasticsearch hybrid search dense retrieval BM25 reranking "
    "evaluation framework NDCG MRR MAP offline online A/B testing recruiter feedback "
    "LLM fine-tuning LoRA QLoRA PEFT learning to rank recommendation system "
    "product company startup scrappy ship fast production deployment real users "
    "Python code quality evaluation benchmark ranking system recommendation system "
    "candidate job description matching recruiter engagement metrics "
    "senior engineer applied scientist NLP 6 8 years product company not consulting "
    "end-to-end ranking search recommendation shipped production scale "
    "hybrid retrieval dense sparse evaluation offline online mentor engineer "
)

# ─────────────────────── REQUIRED SKILLS (from JD "absolutely need") ───── #

REQUIRED_SKILLS = {
    # Embeddings & retrieval (explicitly listed as required)
    "embedding", "embeddings", "sentence-transformers", "sentence transformer",
    "openai embeddings", "bge", "e5", "embedding drift", "retrieval",
    # Vector DBs (explicitly listed)
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "vector database", "vector search", "hybrid search",
    # Evaluation (explicitly listed as required)
    "ndcg", "mrr", "map", "evaluation", "a/b testing", "ab testing",
    "information retrieval", "ranking", "reranking", "bm25",
    "dense retrieval", "semantic search",
    # Python (explicitly listed)
    "python",
}

# "Things we'd like you to have" — preferred but not required
PREFERRED_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "llm",
    "learning to rank", "xgboost", "recommendation", "recommendation system",
    "distributed systems", "inference optimization",
    "pytorch", "transformers", "huggingface", "bert", "gpt",
    "rag", "retrieval augmented", "langchain",
    "lucene", "solr", "redis", "kafka", "spark",
    "deep learning", "neural network", "nlp",
}

# From JD: "explicitly do NOT want" — career/title disqualifiers
HARD_DISQUALIFY_TITLES = [
    "marketing manager", "sales manager", "hr manager", "human resources",
    "accountant", "civil engineer", "operations manager", "finance manager",
    "project manager", "content writer", "graphic designer",
    "customer success", "business development", "supply chain",
    "logistics", "fashion", "interior design",
]

# Weak negative signals (not hard disqualify, but penalise)
SOFT_NEGATIVE_TITLES = [
    "product manager",  # JD says needs to write code, PM doesn't
    "tech lead",        # JD: "not written production code in 18 months"
    "architect",        # Same concern
]

# From JD: "positive" title signals
STRONG_POSITIVE_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "research engineer", "applied scientist",
    "search engineer", "ranking engineer", "recommendation engineer",
    "data scientist",  # borderline but valid
]

MODERATE_POSITIVE_TITLES = [
    "software engineer", "backend engineer", "data engineer",
    "senior engineer", "staff engineer", "principal engineer",
    "deep learning engineer", "computer vision engineer",
    "full stack engineer", "platform engineer",
]

# From JD: consulting companies = explicit disqualifier if ONLY experience
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "l&t infotech", "mindtree",
    "igate", "patni", "niit technologies",
}

# From JD: "CV/Speech/Robotics without NLP/IR" = not a fit
WRONG_DOMAIN_SIGNALS = [
    "computer vision", "speech recognition", "robotics", "autonomous",
    "self-driving", "medical imaging", "satellite imagery",
]

# Location preferences (from JD)
TIER1_LOCATIONS = ["pune", "noida"]
TIER2_LOCATIONS = ["hyderabad", "mumbai", "delhi", "bengaluru", "bangalore",
                   "gurugram", "gurgaon", "ncr"]


# ─────────────────────── HONEYPOT DETECTION ─────────────────────────────── #

def detect_honeypot(candidate: dict) -> bool:
    """
    Detect impossible/fake profiles.
    JD mentions ~80 honeypots in dataset.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Pattern 1: Many "expert" skills claimed but 0 months actual usage
    zero_month_experts = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if zero_month_experts >= 5:
        return True

    # Pattern 2: Stated YoE wildly inconsistent with career history
    stated_yoe = profile.get("years_of_experience", 0)
    if career:
        total_months = sum(c.get("duration_months", 0) for c in career)
        actual_yoe = total_months / 12
        if actual_yoe > 0 and stated_yoe > actual_yoe * 3.5:
            return True

    # Pattern 3: 40+ skills all claimed as expert (impossible breadth)
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 20:
        return True

    return False


# ─────────────────────── SCORING FUNCTIONS ──────────────────────────────── #

def score_skills(candidate: dict) -> float:
    """
    0–100. Weights quality over quantity.

    JD explicitly warns about keyword stuffing. We counter this by:
    - Requiring proficiency >= intermediate for required skills to count
    - Weighting duration_months (real usage vs listed)
    - Weighting endorsements (social validation)
    - Platform assessment scores (objective test results)
    """
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    if not skills:
        return 0.0

    PROF_MAP = {"beginner": 0.1, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}

    req_score = 0.0
    pref_score = 0.0
    assess_score = 0.0

    for skill in skills:
        name_l = skill.get("name", "").lower().strip()
        prof = PROF_MAP.get(skill.get("proficiency", "beginner"), 0.1)
        endorsements = min(skill.get("endorsements", 0), 100)
        duration = min(skill.get("duration_months", 0), 60)

        is_req = any(r in name_l or name_l in r for r in REQUIRED_SKILLS)
        is_pref = (not is_req) and any(p in name_l or name_l in p for p in PREFERRED_SKILLS)

        if is_req:
            # Quality multiplier: proficiency * duration weight * endorsement weight
            quality = prof * (0.5 + 0.5 * min(duration / 24, 1.0))
            social = 1.0 + min(endorsements / 50, 0.5)
            req_score += quality * social * 12  # up to 12 per skill
        elif is_pref:
            quality = prof * (0.5 + 0.5 * min(duration / 12, 1.0))
            pref_score += quality * 4

    # Platform skill assessment scores (objective, harder to fake)
    for skill_name, sc in assessments.items():
        n = skill_name.lower()
        if any(r in n or n in r for r in REQUIRED_SKILLS):
            assess_score += sc * 0.5
        elif any(p in n or n in p for p in PREFERRED_SKILLS):
            assess_score += sc * 0.2

    # Wrong domain penalty (CV/Speech/Robotics without NLP)
    wrong_domain_skills = sum(
        1 for s in skills
        if any(w in s.get("name", "").lower() for w in WRONG_DOMAIN_SIGNALS)
    )
    nlp_ir_skills = sum(
        1 for s in skills
        if any(r in s.get("name", "").lower() for r in
               ["nlp", "retrieval", "ranking", "embedding", "search", "recommendation"])
    )
    if wrong_domain_skills >= 2 and nlp_ir_skills == 0:
        req_score *= 0.3  # Strong penalty

    total = min(req_score, 50) + min(pref_score, 30) + min(assess_score, 20)
    return min(float(total), 100.0)


def score_experience(candidate: dict) -> float:
    """
    0–100. This is the most complex score because the JD has very specific
    requirements about WHAT KIND of experience counts.

    Explicit JD disqualifiers:
    - Pure research, no production → hard penalty
    - All-consulting career → hard penalty
    - Title chasers (avg tenure < 18mo) → penalty
    - Not coding for 18mo (pure lead/architect roles) → penalty
    - CV/Speech/Robotics without NLP → penalty
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "").lower()

    # ── 1. YoE Score (0–25) — sweet spot 5-9 from JD ─────────────────── #
    if yoe < 3:
        yoe_score = yoe * 3           # 0–9
    elif yoe < 5:
        yoe_score = 9 + (yoe - 3) * 5  # 9–19
    elif yoe <= 9:
        yoe_score = 19 + (yoe - 5) * 1.5  # 19–25: sweet spot
    else:
        yoe_score = max(25 - (yoe - 9) * 3, 8)  # gradual penalty for 10+

    # ── 2. Title Score (0–30) ─────────────────────────────────────────── #
    title_score = 8  # neutral default

    # Hard disqualify titles (from JD "explicitly do NOT want")
    for neg in HARD_DISQUALIFY_TITLES:
        if neg in current_title:
            title_score = -25
            break
    else:
        # Soft negatives
        for soft in SOFT_NEGATIVE_TITLES:
            if soft in current_title:
                title_score = 2  # very low but not disqualify
                break
        else:
            # Positives
            for strong in STRONG_POSITIVE_TITLES:
                if strong in current_title:
                    title_score = 28
                    # Seniority bonus
                    if any(m in current_title for m in ["senior", "staff", "principal", "lead", "head"]):
                        title_score = 30
                    break
            else:
                for mod in MODERATE_POSITIVE_TITLES:
                    if mod in current_title:
                        title_score = 18
                        if any(m in current_title for m in ["senior", "staff", "principal"]):
                            title_score = 22
                        break

    # ── 3. Career Trajectory Score (0–30) ────────────────────────────── #
    traj_score = 0
    if career:
        product_roles = 0
        consulting_roles = 0
        ai_ml_roles = 0
        production_signals = 0
        total_ai_months = 0

        for job in career:
            company_l = job.get("company", "").lower()
            job_title_l = job.get("title", "").lower()
            job_desc_l = job.get("description", "").lower()
            duration = job.get("duration_months", 0)

            # Company type
            is_consulting = any(c in company_l for c in CONSULTING_COMPANIES)
            if is_consulting:
                consulting_roles += 1
            else:
                product_roles += 1

            # AI/ML role?
            ai_title_signals = ["ml", "ai ", "nlp", "data scien", "search",
                                 "ranking", "retrieval", "machine learning",
                                 "deep learning", "applied scientist", "research engineer"]
            is_ai_role = any(t in job_title_l for t in ai_title_signals)

            # Production signals in description
            prod_signals = ["deployed", "production", "served", "real users",
                            "at scale", "million", "billion", "latency", "throughput",
                            "a/b test", "shipped", "launched", "improved"]
            has_prod = any(p in job_desc_l for p in prod_signals)
            if has_prod:
                production_signals += 1

            if is_ai_role:
                ai_ml_roles += 1
                total_ai_months += duration

        # Scoring
        # Product company experience (JD explicitly values this)
        traj_score += min(product_roles * 6, 15)
        # AI/ML roles
        traj_score += min(ai_ml_roles * 5, 15)
        # Production deployment signals
        traj_score += min(production_signals * 3, 9)

        # HARD PENALTY: All-consulting career (explicit JD disqualifier)
        if consulting_roles > 0 and product_roles == 0:
            traj_score -= 30  # very strong penalty

        traj_score = max(0.0, min(traj_score, 30.0))

    # ── 4. Job Stability Score (0–15) ────────────────────────────────── #
    # JD: "we need someone who plans to be here 3+ years"
    # "title-chasers switching every 1.5 years are a red flag"
    stability_score = 0
    if len(career) >= 2:
        tenures = [j.get("duration_months", 0) for j in career]
        avg_tenure = sum(tenures) / len(tenures)
        if avg_tenure >= 36:
            stability_score = 15
        elif avg_tenure >= 24:
            stability_score = 12
        elif avg_tenure >= 18:
            stability_score = 8
        elif avg_tenure >= 12:
            stability_score = 4
        else:
            stability_score = 0  # Job hopper: JD explicitly calls this out

    total = yoe_score + title_score + traj_score + stability_score
    return max(0.0, min(float(total), 100.0))


def score_availability(candidate: dict) -> float:
    """
    0–100. The JD is EXPLICIT about this:
    "A perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% recruiter response rate is, for hiring purposes,
    not actually available. Down-weight them appropriately."

    This is a first-class signal, not an afterthought.
    """
    s = candidate.get("redrob_signals", {})

    # ── Open to work (0–20) — clearest availability signal ──────────── #
    otw_score = 20 if s.get("open_to_work_flag", False) else 0

    # ── Last active recency (0–25) — "hasn't logged in for 6 months" ── #
    active_score = 0
    last_active_str = s.get("last_active_date", "")
    if last_active_str:
        try:
            la = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (TODAY - la).days
            if days_inactive <= 7:
                active_score = 25
            elif days_inactive <= 14:
                active_score = 22
            elif days_inactive <= 30:
                active_score = 18
            elif days_inactive <= 60:
                active_score = 12
            elif days_inactive <= 90:
                active_score = 7
            elif days_inactive <= 180:
                active_score = 3
            else:
                active_score = 0  # 6+ months: JD explicitly says down-weight
        except ValueError:
            active_score = 5

    # ── Notice period (0–20) — JD: "love sub-30 day, can buy out 30" ── #
    notice = s.get("notice_period_days", 90)
    if notice == 0:
        notice_score = 20     # Immediately available
    elif notice <= 15:
        notice_score = 19
    elif notice <= 30:
        notice_score = 16     # JD: "sub-30 is ideal, can buy out 30"
    elif notice <= 60:
        notice_score = 7
    elif notice <= 90:
        notice_score = 3
    else:
        notice_score = 0      # JD: "90+ day bar gets higher"

    # ── Recruiter engagement (0–20) — "5% response rate = not available" #
    response_rate = s.get("recruiter_response_rate", 0.5)
    response_time = s.get("avg_response_time_hours", 48)

    # Response rate is the key signal the JD calls out
    if response_rate <= 0.05:
        engagement_score = 0   # JD explicitly says this person is not available
    elif response_rate <= 0.2:
        engagement_score = 5
    elif response_rate <= 0.5:
        engagement_score = 10
    elif response_rate <= 0.8:
        engagement_score = 15
    else:
        engagement_score = 20

    # Response time modifier
    if response_time <= 4:
        engagement_score = min(engagement_score + 3, 20)
    elif response_time <= 24:
        engagement_score = min(engagement_score + 1, 20)
    elif response_time > 72:
        engagement_score = max(engagement_score - 3, 0)

    # ── Interview & offer behavior (0–15) ────────────────────────────── #
    completion = s.get("interview_completion_rate", 0.5)
    acceptance = s.get("offer_acceptance_rate", -1)
    behav_score = completion * 10
    if acceptance >= 0:
        behav_score += acceptance * 5

    total = otw_score + active_score + notice_score + engagement_score + behav_score
    return max(0.0, min(float(total), 100.0))


def score_platform_trust(candidate: dict) -> float:
    """
    0–100. Platform signals for profile authenticity and market demand.
    Also captures the JD hint: "Active on Redrob platform...
    so we can actually talk to them."
    """
    s = candidate.get("redrob_signals", {})

    # Profile completeness (0–25)
    completeness = s.get("profile_completeness_score", 0) * 0.25

    # Verification signals (0–15)
    verification = (
        (5 if s.get("verified_email", False) else 0) +
        (5 if s.get("verified_phone", False) else 0) +
        (5 if s.get("linkedin_connected", False) else 0)
    )

    # GitHub (0–20): very relevant for this AI engineering role
    github = s.get("github_activity_score", -1)
    # JD: "open source contributions in the AI/ML space" = preferred
    if github == -1:
        github_score = 0
    elif github >= 70:
        github_score = 20  # Strong open-source signal
    elif github >= 40:
        github_score = 12
    else:
        github_score = github * 0.15

    # Platform engagement/demand (0–25)
    saved = min(s.get("saved_by_recruiters_30d", 0), 30)
    searches = min(s.get("search_appearance_30d", 0), 500)
    connections = min(s.get("connection_count", 0), 500)
    endorsements_rx = min(s.get("endorsements_received", 0), 100)
    demand = min((saved / 30) * 12 + (searches / 500) * 7, 19)
    social = min((connections / 500) * 4 + (endorsements_rx / 100) * 2, 6)

    # Skill assessment scores (0–15): objective platform-verified skills
    assess = s.get("skill_assessment_scores", {})
    assess_bonus = 0
    for skill_name, sc in assess.items():
        n = skill_name.lower()
        if any(r in n or n in r for r in REQUIRED_SKILLS):
            assess_bonus += sc * 0.15
    assess_bonus = min(assess_bonus, 15)

    total = completeness + verification + github_score + demand + social + assess_bonus
    return max(0.0, min(float(total), 100.0))


def score_location(candidate: dict) -> float:
    """
    0–100. JD is specific: Pune/Noida preferred.
    Hyderabad/Mumbai/Delhi NCR/Bangalore also welcome.
    Outside India: case-by-case, no visa sponsorship.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    will_relocate = signals.get("willing_to_relocate", False)

    if "india" not in country and not will_relocate:
        return 5.0  # JD: no visa sponsorship, case-by-case

    if any(c in location for c in TIER1_LOCATIONS):
        return 100.0   # Perfect match
    if any(c in location for c in TIER2_LOCATIONS):
        return 78.0    # "Welcome to apply"
    if "india" in country and will_relocate:
        return 58.0    # Can relocate from India
    if "india" in country:
        return 32.0    # In India but may not relocate
    return 10.0


# ─────────────────────── TEXT BUILDER ──────────────────────────────────── #

def build_candidate_text(candidate: dict) -> str:
    """
    Build dense text for TF-IDF. Weight production and AI/ML signals more
    by repeating key terms from relevant experience.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    certs = candidate.get("certifications", [])

    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]

    # Skills: weight required/advanced skills by repeating them
    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        name_l = name.lower()

        is_req = any(r in name_l or name_l in r for r in REQUIRED_SKILLS)
        if is_req and prof in ("advanced", "expert"):
            parts.extend([name, name, name])  # repeat 3x for TF-IDF weight
        elif is_req:
            parts.extend([name, name])
        else:
            parts.append(f"{name} {prof}")

    # Career: focus on descriptions for production signals
    for job in career[:6]:
        parts.append(job.get("title", ""))
        parts.append(job.get("company", ""))
        desc = job.get("description", "")
        # Full description for AI/ML roles, truncated for others
        job_title_l = job.get("title", "").lower()
        if any(t in job_title_l for t in ["ml", "ai", "nlp", "data", "search", "engineer"]):
            parts.append(desc[:500])
        else:
            parts.append(desc[:100])

    for cert in certs[:4]:
        parts.append(cert.get("name", ""))

    return " ".join(p for p in parts if p).strip()


# ─────────────────────── REASONING BUILDER ─────────────────────────────── #

def build_reasoning(candidate: dict, scores: dict, final_score: float) -> str:
    """
    Grounded, specific reasoning. Mentions actual facts from the profile.
    Mirrors how a recruiter would explain the ranking to a hiring manager.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")
    notice = signals.get("notice_period_days", 90)
    otw = signals.get("open_to_work_flag", False)
    github = signals.get("github_activity_score", -1)
    last_active_str = signals.get("last_active_date", "")
    response_rate = signals.get("recruiter_response_rate", 0)
    preferred_mode = signals.get("preferred_work_mode", "")

    # Required skills this candidate actually has (advanced/expert)
    strong_req_skills = [
        s["name"] for s in skills
        if any(r in s["name"].lower() for r in REQUIRED_SKILLS)
        and s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 0) >= 6
    ][:5]

    # AI/ML jobs in career
    ai_jobs = [
        j for j in career
        if any(t in j.get("title", "").lower() for t in
               ["ml", "ai ", "nlp", "data scien", "search", "ranking",
                "retrieval", "machine learning", "deep learning", "applied"])
    ]

    # Consulting-only check
    consulting_only = career and all(
        any(c in j.get("company", "").lower() for c in CONSULTING_COMPANIES)
        for j in career
    )

    # Build reasoning
    parts = []

    # Opening: title, YoE, location
    parts.append(f"{yoe:.0f}-year {title} based in {location}.")

    # Skill evidence
    if strong_req_skills:
        parts.append(f"Relevant skills: {', '.join(strong_req_skills)}.")
    elif scores.get("skill", 0) < 15:
        parts.append("Weak overlap with required retrieval/embedding/ranking skills.")

    # Career trajectory
    if ai_jobs:
        recent = ai_jobs[0]
        parts.append(
            f"{len(ai_jobs)} AI/ML role(s) including {recent.get('title','')} "
            f"at {recent.get('company','')}."
        )
    if consulting_only:
        parts.append("All-consulting career — explicit JD disqualifier.")

    # Availability
    if not otw:
        parts.append("Not marked open-to-work.")
    if notice <= 30:
        parts.append(f"Short notice period ({notice}d).")
    elif notice > 60:
        parts.append(f"Long notice period ({notice}d).")

    # Activity
    if last_active_str:
        try:
            la = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days = (TODAY - la).days
            if days > 180:
                parts.append(f"Inactive for {days} days — JD says to down-weight.")
            elif days <= 14:
                parts.append("Active on platform recently.")
        except ValueError:
            pass

    if response_rate <= 0.05:
        parts.append(f"5% recruiter response rate — JD flag: not actually available.")
    elif response_rate < 0.2:
        parts.append(f"Low recruiter response rate ({response_rate:.0%}).")

    if github > 70:
        parts.append(f"Strong GitHub activity ({github:.0f}/100) — open-source signal.")

    if preferred_mode:
        parts.append(f"Prefers {preferred_mode} work.")

    return " ".join(parts)


# ─────────────────────── WEIGHTS ────────────────────────────────────────── #

# Based on JD priorities:
# Skills + Experience dominate (technical fit is hard filter)
# Availability is explicitly called out by JD as first-class
# Semantic captures implicit matches the rule-based misses
# Platform and location are tiebreakers
WEIGHTS = {
    "semantic":      0.22,  # TF-IDF cosine sim to JD
    "skill":         0.28,  # Skill quality + JD relevance (highest weight)
    "experience":    0.25,  # Career trajectory — JD has very specific criteria
    "availability":  0.15,  # Explicitly flagged in JD as must-weight
    "platform":      0.05,  # Profile trust + GitHub
    "location":      0.05,  # Pune/Noida preferred
}


# ─────────────────────── MAIN ────────────────────────────────────────────── #

def run_ranking(candidates_path, out_path="output/submission.csv", progress_cb=None):
    """
    Core ranking pipeline, reusable from CLI (main()) or from a UI (Streamlit app.py).

    Args:
        candidates_path: path to the candidates JSONL file
        out_path: where to write submission.csv
        progress_cb: optional callable(message: str) for live progress updates
                     (Streamlit passes st.write or st.status here; CLI leaves it None)

    Returns:
        top100: list of result dicts (candidate_id, final_score, comp, candidate)
        out_path: Path object of the written CSV
        elapsed: total seconds taken
    """
    def log(msg):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    t0 = time.time()

    # ── 1. Load candidates ──────────────────────────────────────────────── #
    log(f"[{time.time()-t0:.1f}s] Reading {candidates_path}...")
    candidates = []
    with open(candidates_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(
                    json.loads(line.decode("utf-8", errors="replace"))
                )
            except json.JSONDecodeError:
                continue
    log(f"[{time.time()-t0:.1f}s] Loaded {len(candidates):,} candidates.")

    # ── 2. Quick pre-filter (100K → ~5K) ───────────────────────────────── #
    log(f"[{time.time()-t0:.1f}s] Pre-filtering...")

    def quick_score(c: dict) -> float:
        if detect_honeypot(c):
            return -9999.0

        p = c.get("profile", {})
        s = c.get("redrob_signals", {})
        title = p.get("current_title", "").lower()
        skills_list = c.get("skills", [])

        score = 0.0

        # Hard disqualify titles
        for neg in HARD_DISQUALIFY_TITLES:
            if neg in title:
                score -= 80
                break

        # Strong title match
        for pos in STRONG_POSITIVE_TITLES:
            if pos in title:
                score += 30
                break
        else:
            for mod in MODERATE_POSITIVE_TITLES:
                if mod in title:
                    score += 15
                    break

        # Availability
        if s.get("open_to_work_flag", False):
            score += 12

        # YoE fit
        yoe = p.get("years_of_experience", 0)
        if 5 <= yoe <= 9:
            score += 15
        elif 3 <= yoe < 5 or 9 < yoe <= 12:
            score += 8

        # Required skill count
        req_count = sum(
            1 for sk in skills_list
            if any(r in sk.get("name", "").lower() for r in REQUIRED_SKILLS)
            and sk.get("proficiency") in ("intermediate", "advanced", "expert")
        )
        score += req_count * 6

        # Last active (rough)
        la = s.get("last_active_date", "")
        if la:
            try:
                days = (TODAY - datetime.strptime(la, "%Y-%m-%d").date()).days
                if days > 365:
                    score -= 20
                elif days <= 30:
                    score += 8
            except ValueError:
                pass

        return score

    scored = sorted(
        range(len(candidates)),
        key=lambda i: quick_score(candidates[i]),
        reverse=True,
    )
    TOP_K = min(5000, len(candidates))
    filtered = [candidates[i] for i in scored[:TOP_K]]
    log(f"[{time.time()-t0:.1f}s] Pre-filtered to {len(filtered):,} candidates.")

    # ── 3. TF-IDF semantic scoring ──────────────────────────────────────── #
    log(f"[{time.time()-t0:.1f}s] Building TF-IDF model...")
    texts = [build_candidate_text(c) for c in filtered]
    all_texts = [JD_TEXT] + texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=60000,
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = tfidf_matrix[0]
    candidate_vecs = tfidf_matrix[1:]
    sem_scores = cosine_similarity(candidate_vecs, jd_vec).flatten()
    log(f"[{time.time()-t0:.1f}s] TF-IDF done. Max sim={sem_scores.max():.3f}")

    # ── 4. Full multi-dimensional scoring ──────────────────────────────── #
    log(f"[{time.time()-t0:.1f}s] Scoring {len(filtered):,} candidates...")
    results = []
    for candidate, sem in zip(filtered, sem_scores):
        if detect_honeypot(candidate):
            continue

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
            "final_score":  max(0.0, min(100.0, final)),
            "comp":         comp,
            "candidate":    candidate,
        })

    # ── 5. Sort: score desc, tie-break candidate_id asc ────────────────── #
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top100 = results[:100]

    log(f"[{time.time()-t0:.1f}s] "
        f"Top score={top100[0]['final_score']:.2f}, "
        f"#100 score={top100[-1]['final_score']:.2f}")

    # ── 6. Normalise to 0.40–0.99 (non-increasing, tie-break safe) ─────── #
    max_s = top100[0]["final_score"]
    min_s = top100[-1]["final_score"]
    rng = max_s - min_s if max_s > min_s else 1.0

    # ── 7. Write CSV ────────────────────────────────────────────────────── #
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_norm = None
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top100, 1):
            norm = 0.40 + ((r["final_score"] - min_s) / rng) * 0.59
            norm = round(norm, 4)
            # Guarantee non-increasing (float rounding edge case guard)
            if prev_norm is not None and norm > prev_norm:
                norm = prev_norm
            prev_norm = norm

            reasoning = build_reasoning(r["candidate"], r["comp"], r["final_score"])
            writer.writerow([r["candidate_id"], rank, norm, reasoning])

    elapsed = time.time() - t0
    log(f"\n✅ Done in {elapsed:.1f}s  →  {out_path}")
    log("\nTop 15 ranked candidates:")
    log(f"  {'Rank':>4}  {'Candidate ID':<14}  {'Title':<38}  {'YoE':>3}  {'Loc':<22}  Score")
    log("  " + "─" * 100)
    for i, r in enumerate(top100[:15]):
        p = r["candidate"]["profile"]
        log(
            f"  #{i+1:>3}  {r['candidate_id']:<14}  "
            f"{p.get('current_title','')[:38]:<38}  "
            f"{p.get('years_of_experience',0):>3.0f}  "
            f"{p.get('location','')[:22]:<22}  "
            f"{r['final_score']:.2f}"
        )

    # Component score breakdown for top 5
    log("\nComponent breakdown for top 5:")
    log(f"  {'ID':<14}  {'Sem':>5}  {'Skill':>5}  {'Exp':>5}  {'Avail':>5}  {'Plat':>5}  {'Loc':>5}  {'Final':>6}")
    log("  " + "─" * 70)
    for r in top100[:5]:
        c = r["comp"]
        log(
            f"  {r['candidate_id']:<14}  "
            f"{c['semantic']:>5.1f}  {c['skill']:>5.1f}  "
            f"{c['experience']:>5.1f}  {c['availability']:>5.1f}  "
            f"{c['platform']:>5.1f}  {c['location']:>5.1f}  "
            f"{r['final_score']:>6.2f}"
        )

    return top100, out_path, elapsed


def main():
    """Thin CLI wrapper around run_ranking() — unchanged behaviour from before."""
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranker — CPU only, no API calls"
    )
    parser.add_argument(
        "--candidates",
        default="/mnt/user-data/uploads/sample_candidates.json",
    )
    parser.add_argument("--out", default="output/submission.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run_ranking(args.candidates, args.out)


if __name__ == "__main__":
    main()