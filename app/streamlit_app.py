"""
streamlit_app.py
AI-Based Resume Screening and Candidate Ranking System - Streamlit dashboard.
Run from the project root:  streamlit run app/streamlit_app.py
"""
import hashlib
import io
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))  # lets "from src..." work locally and on Streamlit Cloud

from src.matcher import rank_candidates
from src.parser import parse_resume
from src.skills import extract_jd_skills, get_all_skill_names

SAMPLE_JD_PATH = ROOT / "data" / "sample_jd" / "data_scientist_jd.txt"

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
    st.caption("Resumes are processed in memory and are not stored.")

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
        df[["Rank", "Candidate", "Text Similarity (%)", "Skill Match (%)",
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

st.divider()
st.caption("Scores are similarity indicators to support a recruiter. "
           "They are not a hiring verdict, and the final decision always stays with a human.")