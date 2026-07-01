import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from rank import run_ranking, build_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide", page_icon="🎯")

st.title("🎯 Redrob Intelligent Candidate Ranking System")
st.caption("Multi-dimensional AI scoring | CPU-Only | No API Calls | Senior AI Engineer Role")

st.markdown("""
This live demo runs the exact same `rank.py` pipeline submitted for the 
**India Runs Data & AI Challenge — Redrob AI Candidate Ranking Problem**.  
Upload a `.jsonl` candidates file or use the bundled 300-row sample to see live results.
""")

DEFAULT_SAMPLE = Path(__file__).parent / "sample_candidates.jsonl"

col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload candidates.jsonl (optional — leave blank to use bundled sample)",
        type=["jsonl", "json"],
    )
with col2:
    st.markdown("### Scoring Weights")
    weights_data = {
        "Dimension": ["Skill Quality", "Experience", "Semantic (TF-IDF)", "Availability", "Platform Trust", "Location"],
        "Weight": [28, 25, 22, 15, 5, 5],
    }
    wdf = pd.DataFrame(weights_data)
    st.dataframe(wdf, hide_index=True, use_container_width=True)

run_clicked = st.button("🚀 Run Ranking", type="primary", use_container_width=True)

if run_clicked:
    if uploaded is not None:
        tmp_path = Path("uploaded_candidates.jsonl")
        tmp_path.write_bytes(uploaded.getvalue())
        candidates_path = tmp_path
        source_label = f"uploaded file: {uploaded.name}"
    elif DEFAULT_SAMPLE.exists():
        candidates_path = DEFAULT_SAMPLE
        source_label = "bundled sample dataset (300 candidates)"
    else:
        st.error("No candidates file found. Please upload a .jsonl file.")
        st.stop()

    st.info(f"Running ranking on {source_label}...")
    progress_box = st.empty()
    log_lines = []

    def progress_cb(msg):
        log_lines.append(msg)
        progress_box.code("\n".join(log_lines[-10:]))

    with st.spinner("Scoring candidates — this may take a few seconds..."):
        top100, out_path, elapsed = run_ranking(
            str(candidates_path),
            out_path="output/submission.csv",
            progress_cb=progress_cb,
        )

    progress_box.empty()
    st.success(f"✅ Done in {elapsed:.1f} seconds — ranked {len(top100)} candidates.")

    # ── Build display dataframe ──
    rows = []
    for i, r in enumerate(top100[:15], start=1):
        p = r["candidate"]["profile"]
        c = r["comp"]
        rows.append({
            "Rank": i,
            "Candidate ID": r["candidate_id"],
            "Title": p.get("current_title", ""),
            "YoE": int(p.get("years_of_experience", 0)),
            "Location": p.get("location", ""),
            "Score": round(r["final_score"], 2),
            "Skill": round(c["skill"], 1),
            "Experience": round(c["experience"], 1),
            "Semantic": round(c["semantic"], 1),
            "Availability": round(c["availability"], 1),
            "Platform": round(c["platform"], 1),
            "Location Score": round(c["location"], 1),
        })
    df = pd.DataFrame(rows)

    st.divider()

    # ── Tab layout ──
    tab1, tab2, tab3 = st.tabs(["🏆 Top 15 Ranked", "📊 Score Breakdown Charts", "📝 Reasoning"])

    with tab1:
        st.subheader("Top 15 Candidates")
        display_cols = ["Rank", "Candidate ID", "Title", "YoE", "Location", "Score"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        csv_bytes = Path(out_path).read_bytes()
        st.download_button(
            "⬇️ Download full submission.csv (top 100)",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
        )

    with tab2:
        st.subheader("Multi-Dimensional Score Breakdown — Top 10")

        # Bar chart: overall scores
        fig1 = go.Figure(go.Bar(
            x=df["Candidate ID"][:10],
            y=df["Score"][:10],
            marker_color="#4F8EF7",
            text=df["Score"][:10],
            textposition="outside",
        ))
        fig1.update_layout(
            title="Overall Final Score — Top 10 Candidates",
            xaxis_title="Candidate ID",
            yaxis_title="Final Score (0–100)",
            height=380,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Grouped bar chart: dimension breakdown for top 5
        dims = ["Skill", "Experience", "Semantic", "Availability", "Platform", "Location Score"]
        colors = ["#4F8EF7", "#F76F4F", "#4FD7A0", "#F7C44F", "#A04FF7", "#F74FA0"]

        fig2 = go.Figure()
        for dim, color in zip(dims, colors):
            fig2.add_trace(go.Bar(
                name=dim,
                x=df["Candidate ID"][:5],
                y=df[dim][:5],
                marker_color=color,
            ))
        fig2.update_layout(
            barmode="group",
            title="Component Score Breakdown — Top 5 Candidates",
            xaxis_title="Candidate ID",
            yaxis_title="Score (0–100)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Radar chart for #1 candidate
        st.subheader(f"Radar Profile — Rank #1: {df['Candidate ID'][0]}")
        r1 = top100[0]["comp"]
        categories = ["Skill", "Experience", "Semantic", "Availability", "Platform", "Location"]
        values = [r1["skill"], r1["experience"], r1["semantic"], r1["availability"], r1["platform"], r1["location"]]
        values_norm = [v / 100 * 100 for v in values]

        fig3 = go.Figure(go.Scatterpolar(
            r=values_norm + [values_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(79,142,247,0.3)",
            line=dict(color="#4F8EF7", width=2),
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Recruiter Reasoning — Top 5 Candidates")
        for r in top100[:5]:
            p = r["candidate"]["profile"]
            with st.expander(
                f"#{top100.index(r)+1} — {r['candidate_id']} | {p.get('current_title','')} | Score: {r['final_score']:.2f}"
            ):
                reasoning = build_reasoning(r["candidate"], r["comp"], r["final_score"])
                st.write(reasoning)

                # Mini score bar
                comp = r["comp"]
                mini_df = pd.DataFrame({
                    "Dimension": ["Skill", "Experience", "Semantic", "Availability", "Platform", "Location"],
                    "Score": [
                        round(comp["skill"], 1),
                        round(comp["experience"], 1),
                        round(comp["semantic"], 1),
                        round(comp["availability"], 1),
                        round(comp["platform"], 1),
                        round(comp["location"], 1),
                    ]
                })
                st.bar_chart(mini_df.set_index("Dimension"))

st.divider()
st.caption(
    "Deepanshi Khandelwal | India Runs Data & AI Challenge — Redrob AI Candidate Ranking | "
    "CPU-Only Pipeline | rank.py + app.py"
)