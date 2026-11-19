"""
matcher.py
TF-IDF + Cosine Similarity matching engine, combined scoring and ranking.

Final Score = w_text x Text Similarity (%) + w_skill x Skill Match (%)
Default weights 0.70 / 0.30 are an initial design choice, not a universal rule.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess_text
from src.skills import compare_skills, extract_jd_skills, extract_skills

DEFAULT_TEXT_WEIGHT = 0.70
DEFAULT_SKILL_WEIGHT = 0.30

RESULT_COLUMNS = ["Rank", "Candidate", "File", "Email", "Phone",
                  "Text Similarity (%)", "Skill Match (%)", "Final Score (%)",
                  "Matched Skills", "Missing Skills", "Skills Found", "Status"]


def build_vectorizer(ngram_range=(1, 2)):
    """
    TF-IDF vectorizer that trusts OUR tokens (already cleaned by preprocessing.py).
    tokenizer=str.split keeps tokens like c++, node.js and scikit-learn intact.
    """
    return TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                           lowercase=False, ngram_range=ngram_range)


def cosine_manual(a, b) -> float:
    """Cosine Similarity = (A . B) / (||A|| x ||B||), written out for understanding."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def compute_text_similarity(jd_processed: str, resumes_processed: list,
                            ngram_range=(1, 2)) -> np.ndarray:
    """
    Cosine similarity (0 to 1) between the job description and each resume.
    Both inputs must already be preprocessed (preprocess_text output).
    """
    count = len(resumes_processed)
    if not jd_processed.strip() or not any(d.strip() for d in resumes_processed):
        return np.zeros(count)

    documents = [jd_processed] + list(resumes_processed)   # row 0 = job description
    try:
        matrix = build_vectorizer(ngram_range).fit_transform(documents)
    except ValueError:                                     # empty vocabulary
        return np.zeros(count)

    scores = cosine_similarity(matrix[0], matrix[1:])[0]
    return np.clip(scores, 0.0, 1.0)


def _validate_weights(text_weight: float, skill_weight: float):
    if text_weight < 0 or skill_weight < 0 or abs(text_weight + skill_weight - 1) > 1e-9:
        raise ValueError("Weights must be non-negative and add up to 1.0")


def calculate_final_score(text_percent: float, skill_percent: float,
                          text_weight=DEFAULT_TEXT_WEIGHT,
                          skill_weight=DEFAULT_SKILL_WEIGHT) -> float:
    """Combine the two component scores (both on a 0-100 scale)."""
    _validate_weights(text_weight, skill_weight)
    return round(text_weight * text_percent + skill_weight * skill_percent, 2)


def rank_candidates(resumes: list, jd_text: str, required_skills: list = None,
                    text_weight=DEFAULT_TEXT_WEIGHT,
                    skill_weight=DEFAULT_SKILL_WEIGHT) -> pd.DataFrame:
    """
    resumes: list of dicts returned by parser.parse_resume()
    required_skills: list of skill names; if None they are detected from the JD
    Returns a DataFrame sorted from best to worst candidate.
    """
    _validate_weights(text_weight, skill_weight)
    if not resumes:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    if required_skills is None:
        required_skills = extract_jd_skills(jd_text)["required"]

    jd_processed = preprocess_text(jd_text)
    resumes_processed = [preprocess_text(r["text"]) for r in resumes]
    text_scores = compute_text_similarity(jd_processed, resumes_processed)

    rows = []
    for resume, text_score in zip(resumes, text_scores):
        skills_found = extract_skills(resume["text"])
        skill_result = compare_skills(skills_found, required_skills)
        text_percent = round(float(text_score) * 100, 2)
        skill_percent = skill_result["match_percent"]

        name = resume["name"]
        if not name or name == "Name not detected":
            name = resume["filename"].rsplit(".", 1)[0]

        rows.append({
            "Candidate": name,
            "File": resume["filename"],
            "Email": resume["email"],
            "Phone": resume["phone"],
            "Text Similarity (%)": text_percent,
            "Skill Match (%)": skill_percent,
            "Final Score (%)": calculate_final_score(text_percent, skill_percent,
                                                     text_weight, skill_weight),
            "Matched Skills": ", ".join(skill_result["matched"]),
            "Missing Skills": ", ".join(skill_result["missing"]),
            "Skills Found": ", ".join(skills_found),
            "Status": resume["status"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Final Score (%)", "Skill Match (%)", "Text Similarity (%)"],
                        ascending=False, kind="mergesort").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df[RESULT_COLUMNS]