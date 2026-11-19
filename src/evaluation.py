"""
evaluation.py
Tools to evaluate the ranking honestly:
  - weight_sensitivity : how the ranking changes with different weights
  - rank_correlation   : how similar two rankings are (Spearman)
  - precision_recall_f1_at_k / evaluate_ranking : metrics against HUMAN labels
No accuracy numbers are invented here; metrics need real human-made labels.
"""
import pandas as pd

from src.matcher import rank_candidates

DEFAULT_WEIGHT_PAIRS = [(1.0, 0.0), (0.9, 0.1), (0.7, 0.3),
                        (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]


def weight_sensitivity(resumes, jd_text, weight_pairs=None, required_skills=None):
    """Returns (ranks_table, scores_table); rows = file, columns = 'text/skill' weights."""
    weight_pairs = weight_pairs or DEFAULT_WEIGHT_PAIRS
    ranks, scores = {}, {}
    for text_w, skill_w in weight_pairs:
        df = rank_candidates(resumes, jd_text, required_skills, text_w, skill_w)
        label = f"{round(text_w * 100)}/{round(skill_w * 100)}"
        ranks[label] = df.set_index("File")["Rank"]
        scores[label] = df.set_index("File")["Final Score (%)"]
    return pd.DataFrame(ranks), pd.DataFrame(scores)


def rank_correlation(ranks_table: pd.DataFrame, base_label: str) -> dict:
    """Spearman correlation of every ranking against the base ranking (1.0 = identical)."""
    return {label: round(ranks_table[base_label].corr(ranks_table[label], method="spearman"), 3)
            for label in ranks_table.columns}


def precision_recall_f1_at_k(ranked_files: list, relevant_files: set, k: int) -> dict:
    """
    Precision@k = relevant candidates in top k / k
    Recall@k    = relevant candidates in top k / all relevant candidates
    F1          = 2 x P x R / (P + R)
    """
    top_k = ranked_files[:k]
    hits = sum(1 for f in top_k if f in relevant_files)
    precision = hits / len(top_k) if top_k else 0.0
    recall = hits / len(relevant_files) if relevant_files else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate_ranking(ranked_df: pd.DataFrame, labels: dict, ks=None) -> pd.DataFrame:
    """labels: {filename: 1 (suitable) or 0 (not suitable)} made by a human."""
    files = [f for f in ranked_df["File"] if f in labels]
    relevant = {f for f in files if labels[f] == 1}
    ks = ks or range(1, len(files) + 1)
    rows = [{"k": k, **precision_recall_f1_at_k(files, relevant, k)} for k in ks]
    return pd.DataFrame(rows)