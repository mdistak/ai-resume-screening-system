"""
streamlit_app.py
AI-Based Resume Screening and Candidate Ranking System - Streamlit dashboard.
Run from the project root:  streamlit run app/streamlit_app.py
"""
import hashlib
import io
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # lets "from src..." work locally and on Streamlit Cloud

from src.database import (clear_history, delete_run, get_run_details, get_run_results,
                          init_db, list_runs, save_analysis)
from src.matcher import rank_candidates
from src.parser import parse_resume
from src.skills import extract_jd_skills, get_all_skill_names
from src.visualization import (candidate_labels, close_figure, comparison_chart,
                               score_breakdown_chart, skill_heatmap, skill_matrix,
                               text_vs_skill_chart)

SAMPLE_JD_PATH = ROOT / "data" / "sample_jd" / "data_scientist_jd.txt"

# History can be switched off with the environment variable RESUME_APP_HISTORY=0
# (we will do this on the public deployment, where the database would be shared).
HISTORY_ENABLED = os.environ.get("RESUME_APP_HISTORY", "1") == "1"
if HISTORY_ENABLED:
    try:
        init_db()
    except (OSError, sqlite3.Error):
        HISTORY_ENABLED = False

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")


def load_sample_jd():
    """Button callback: put the sample job description into the text box."""
    st.session_state["jd_text"] = SAMPLE_JD_PATH.read_text(encoding="utf-8")


def parse_uploads(uploaded_files):
    """Parse every uploaded file in memory. Duplicate file names get (2), (3)..."""
    resumes, seen = [], {}
    progress = st.progress(0.0, text="Reading resumes...")
    for index, uploaded in enumerate(uploaded_files, start=1):
        name = uploaded.name
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            stem, _, extension = name.rpartition(".")
            name = f"{stem} ({seen[uploaded.name]}).{extension}"
        resumes.append(parse_resume(io.BytesIO(uploaded.getvalue()), filename=name))
        progress.progress(index / len(uploaded_files),
                          text=f"Reading {name} ({index}/{len(uploaded_files)})")
    progress.empty()
    return resumes


def current_signature(uploaded_files, jd_text, required_skills, text_pct):
    """A fingerprint of the inputs, used to detect stale results."""
    files = tuple((f.name, f.size) for f in uploaded_files or [])
    jd_hash = hashlib.md5(jd_text.encode("utf-8")).hexdigest()
    return (files, jd_hash, tuple(sorted(required_skills)), text_pct)


def render_history():
    """Shows saved analyses from the SQLite database."""
    runs = list_runs()
    if runs.empty:
        st.info("No saved analyses yet. Tick 'Save this analysis to history' "
                "in the sidebar before analysing.")
        return

    overview = runs.copy()
    overview["Weights (text/skill)"] = [f"{round(t * 100)}/{round(s * 100)}"
                                        for t, s in zip(runs["text_weight"], runs["skill_weight"])]
    overview = overview.rename(columns={
        "run_id": "Run", "created_at": "Date", "job_title": "Job",
        "candidate_count": "Candidates", "top_candidate": "Top candidate",
        "top_score": "Top score (%)"})
    st.dataframe(overview[["Run", "Date", "Job", "Candidates", "Weights (text/skill)",
                           "Top candidate", "Top score (%)"]], hide_index=True)

    options = {f"Run #{row.run_id}  |  {row.created_at}  |  {row.job_title}": int(row.run_id)
               for row in runs.itertuples()}
    chosen = st.selectbox("Open a saved run", list(options))
    run_id = options[chosen]
    details = get_run_details(run_id)
    if details is None:
        return
    saved = get_run_results(run_id)

    st.caption(f"Weights: text {round(details['text_weight'] * 100)}% / "
               f"skill {round(details['skill_weight'] * 100)}%  |  "
               f"Required skills: {details['required_skills'] or 'none'}")
    st.dataframe(saved, hide_index=True)
    st.download_button("Download this run as CSV",
                       data=saved.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"history_run_{run_id}.csv", mime="text/csv",
                       key=f"download_run_{run_id}")

    delete_col, clear_col = st.columns(2)
    if delete_col.button("Delete this run"):
        delete_run(run_id)
        st.rerun()
    confirm = clear_col.checkbox("I want to clear ALL saved history")
    if clear_col.button("Clear all history", disabled=not confirm):
        clear_history()
        st.rerun()


# ------------------------------------------------------------------ header
st.title("AI-Based Resume Screening and Candidate Ranking System")
st.caption("NLP + Machine Learning (TF-IDF and Cosine Similarity). "
           "A recruitment decision-support tool, not an automatic hiring or rejection system.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    text_pct = st.slider("Text similarity weight (%)", 0, 100, 70, step=5)
    skill_pct = 100 - text_pct
    st.write(f"Skill match weight: **{skill_pct}%**")
    st.caption("Final Score = text weight x Text Similarity + skill weight x Skill Match. "
               "The default 70/30 is an initial design choice, not a universal rule.")
    st.divider()
    st.button("Load sample job description", on_click=load_sample_jd)
    st.divider()
    save_history = False
    if HISTORY_ENABLED:
        save_history = st.checkbox(
            "Save this analysis to history", value=False,
            help="Stores scores, skills and candidate names in a local SQLite file. "
                 "Resume text, emails and phone numbers are NOT stored.")
    st.caption("Resume files are processed in memory and never stored. "
               "Saved history keeps scores and skills only.")

# ------------------------------------------------------------------ inputs
st.subheader("1. Upload resumes")
uploaded_files = st.file_uploader("PDF or DOCX files (you can select many)",
                                  type=["pdf", "docx"], accept_multiple_files=True)

st.subheader("2. Job description")
jd_text = st.text_area("Paste the job description here", key="jd_text", height=220)

st.subheader("3. Required skills")
detected = (extract_jd_skills(jd_text) if jd_text.strip()
            else {"required": [], "optional": []})
# a new widget key per JD text, so the detected default refreshes when the JD changes
skills_key = "required_" + hashlib.md5(jd_text.encode("utf-8")).hexdigest()[:8]
required_skills = st.multiselect(
    "Skills used for the Skill Match score (detected automatically, edit if needed)",
    options=get_all_skill_names(), default=detected["required"], key=skills_key)
if detected["optional"]:
    st.caption("Optional skills found in the JD (not scored): " + ", ".join(detected["optional"]))
if jd_text.strip() and not required_skills:
    st.warning("No required skills selected. Skill Match will be 0 for every candidate "
               "and only text similarity will count. Add skills above if needed.")

# ------------------------------------------------------------------ analysis
if st.button("Analyze candidates", type="primary"):
    if not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not jd_text.strip():
        st.error("Please enter a job description.")
    else:
        resumes = parse_uploads(uploaded_files)
        with st.spinner("Calculating TF-IDF and cosine similarity..."):
            ranked = rank_candidates(resumes, jd_text, required_skills,
                                     text_pct / 100, skill_pct / 100)
        st.session_state["results"] = {
            "df": ranked,
            "required": list(required_skills),
            "text_pct": text_pct,
            "skill_pct": skill_pct,
            "time": datetime.now().strftime("%d %b %Y, %H:%M:%S"),
            "signature": current_signature(uploaded_files, jd_text, required_skills, text_pct),
        }
        if save_history:
            saved_run_id = save_analysis(ranked, jd_text, text_pct / 100, skill_pct / 100,
                                         required_skills)
            st.success(f"Analysis saved to history as run #{saved_run_id}.")

# ------------------------------------------------------------------ results
results = st.session_state.get("results")
if results:
    df = results["df"]
    st.divider()
    st.header("Ranking results")

    if results["signature"] != current_signature(uploaded_files, jd_text, required_skills, text_pct):
        st.info("Inputs or settings changed since the last analysis. "
                "Click 'Analyze candidates' to update the results.")

    st.caption(f"Weights: text {results['text_pct']}% / skill {results['skill_pct']}%  |  "
               f"Required skills ({len(results['required'])}): "
               f"{', '.join(results['required']) or 'none'}  |  Generated: {results['time']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates analysed", len(df))
    col2.metric("Top candidate", df.loc[0, "Candidate"])
    col3.metric("Top final score", f"{df.loc[0, 'Final Score (%)']}%")
    col4.metric("Average final score", f"{df['Final Score (%)'].mean():.2f}%")

    for _, row in df[df["Status"] != "OK"].iterrows():
        st.warning(f"{row['File']}: {row['Status']}")

    st.dataframe(
        df[["Rank", "Candidate", "File", "Text Similarity (%)", "Skill Match (%)",
            "Final Score (%)", "Matched Skills", "Missing Skills", "Email", "Phone"]],
        hide_index=True,
        column_config={
            "Text Similarity (%)": st.column_config.ProgressColumn(
                "Text Similarity (%)", min_value=0, max_value=100, format="%.2f"),
            "Skill Match (%)": st.column_config.ProgressColumn(
                "Skill Match (%)", min_value=0, max_value=100, format="%.2f"),
            "Final Score (%)": st.column_config.ProgressColumn(
                "Final Score (%)", min_value=0, max_value=100, format="%.2f"),
        },
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download CSV report", data=csv_bytes,
                       file_name=f"candidate_ranking_{datetime.now():%Y%m%d_%H%M%S}.csv",
                       mime="text/csv")

    st.subheader("Visual analysis")
    tab_breakdown, tab_heatmap, tab_scatter, tab_compare = st.tabs(
        ["Score breakdown", "Skill heatmap", "Text vs skill", "Compare candidates"])

    with tab_breakdown:
        fig = score_breakdown_chart(df, results["text_pct"] / 100, results["skill_pct"] / 100)
        st.pyplot(fig)
        close_figure(fig)
        st.caption("Each bar is the final score split into its two weighted parts. "
                   "The top 15 candidates are shown.")

    with tab_heatmap:
        fig = skill_heatmap(df, results["required"])
        if fig is None:
            st.info("No required skills are selected, so there is nothing to show.")
        else:
            st.pyplot(fig)
            close_figure(fig)
            st.caption("Green = the resume contains the required skill, red = it is missing. "
                       "The top 15 candidates are shown.")

    with tab_scatter:
        fig = text_vs_skill_chart(df)
        st.pyplot(fig)
        close_figure(fig)
        st.caption("Read both scores together. A resume high on skill match but low on text "
                   "similarity (top left) is worth a manual look, for example a long skill "
                   "list with little description of actual work.")

    with tab_compare:
        options = candidate_labels(df)
        picked = st.multiselect("Choose 2 to 4 candidates to compare", options=options,
                                default=options[:min(3, len(options))], max_selections=4)
        if len(picked) < 2:
            st.info("Select at least two candidates.")
        else:
            subset = df[[label in picked for label in options]]
            fig = comparison_chart(subset)
            st.pyplot(fig)
            close_figure(fig)
            if results["required"]:
                st.markdown("**Required skills side by side** (\u2713 matched, \u2717 missing)")
                table = skill_matrix(subset, results["required"]).T
                st.dataframe(table.apply(lambda col: col.map({1: "\u2713", 0: "\u2717"})))

    st.subheader("Candidate details")
    for _, row in df.iterrows():
        with st.expander(f"#{row['Rank']}  {row['Candidate']}  |  Final score {row['Final Score (%)']}%"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Final score", f"{row['Final Score (%)']}%")
            c2.metric("Text similarity", f"{row['Text Similarity (%)']}%")
            c3.metric("Skill match", f"{row['Skill Match (%)']}%")
            st.markdown(f"**Matched skills:** :green[{row['Matched Skills'] or '-'}]")
            st.markdown(f"**Missing skills:** :red[{row['Missing Skills'] or '-'}]")
            st.markdown(f"**All skills found:** {row['Skills Found'] or '-'}")
            st.caption(f"Email: {row['Email']}  |  Phone: {row['Phone']}  |  File: {row['File']}")

# ------------------------------------------------------------------ history
if HISTORY_ENABLED:
    st.divider()
    with st.expander("Analysis history (saved runs)"):
        render_history()

st.divider()
st.caption("Scores are similarity indicators to support a recruiter. "
           "They are not a hiring verdict, and the final decision always stays with a human.")