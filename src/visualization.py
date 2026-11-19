"""
visualization.py
Matplotlib / Seaborn charts for the ranking dashboard.
Every function takes the DataFrame returned by matcher.rank_candidates()
and returns a matplotlib Figure (or None when there is nothing to draw).
"""
import os

import matplotlib

matplotlib.use("Agg")  # draw to memory: works inside Streamlit and in tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

COLOR_TEXT = "#4C78A8"
COLOR_SKILL = "#F58518"
COLOR_MATCHED = "#A8D5BA"
COLOR_MISSING = "#F2B8B5"


def _short(text, limit=22):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def candidate_labels(df):
    """Unique readable labels like '#3 Rahul Sharma' (the rank keeps them unique)."""
    return [f"#{rank} {_short(name)}" for rank, name in zip(df["Rank"], df["Candidate"])]


def skill_matrix(df, required_skills):
    """Candidates x required skills table with 1 (matched) or 0 (missing)."""
    rows = []
    for _, row in df.iterrows():
        matched = {s.strip() for s in str(row["Matched Skills"]).split(",") if s.strip()}
        rows.append([1 if skill in matched else 0 for skill in required_skills])
    return pd.DataFrame(rows, index=candidate_labels(df), columns=list(required_skills))


def score_breakdown_chart(df, text_weight, skill_weight, top_n=15):
    """Stacked horizontal bars: final score = weighted text part + weighted skill part."""
    data = df.head(top_n).iloc[::-1]               # reversed so rank 1 is drawn on top
    labels = candidate_labels(data)
    text_part = data["Text Similarity (%)"].to_numpy() * text_weight
    skill_part = data["Skill Match (%)"].to_numpy() * skill_weight
    totals = text_part + skill_part
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.5 * len(data) + 1.5)))
    ax.barh(y, text_part, color=COLOR_TEXT, label=f"Text similarity x {text_weight:.2f}")
    ax.barh(y, skill_part, left=text_part, color=COLOR_SKILL,
            label=f"Skill match x {skill_weight:.2f}")
    for position, total in zip(y, totals):
        ax.text(total + 0.6, position, f"{total:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(10.0, totals.max() * 1.15))
    ax.set_xlabel("Final score (%)")
    ax.set_title("Score breakdown: weighted text similarity + weighted skill match")
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def skill_heatmap(df, required_skills, top_n=15):
    """Green = required skill found in the resume, red = missing."""
    if not required_skills or df.empty:
        return None
    matrix = skill_matrix(df.head(top_n), required_skills)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.95 * len(required_skills) + 3),
                                    max(3.0, 0.5 * len(matrix) + 1.8)))
    sns.heatmap(matrix, cmap=ListedColormap([COLOR_MISSING, COLOR_MATCHED]),
                vmin=0, vmax=1, cbar=False, linewidths=1.5, linecolor="white",
                annot=np.where(matrix.to_numpy() == 1, "\u2713", "\u2717"), fmt="",
                annot_kws={"size": 10}, ax=ax)
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=40, ha="left")
    ax.set_xlabel("Required skills")
    ax.legend(handles=[Patch(color=COLOR_MATCHED, label="Matched"),
                       Patch(color=COLOR_MISSING, label="Missing")],
              loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    return fig


def text_vs_skill_chart(df, top_n=25):
    """Scatter of text similarity against skill match, one dot per candidate."""
    data = df.head(top_n)
    x = data["Text Similarity (%)"].to_numpy()
    y = data["Skill Match (%)"].to_numpy()
    x_max = max(10.0, x.max() * 1.25)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=80, color=COLOR_TEXT, edgecolor="white", zorder=3)
    for (_, row), xi, yi in zip(data.iterrows(), x, y):
        ax.annotate(f"#{row['Rank']} {_short(row['Candidate'], 16)}", (xi, yi),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(np.median(x), ls="--", color="grey", lw=0.8)
    ax.axhline(np.median(y), ls="--", color="grey", lw=0.8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-5, 108)
    ax.set_xlabel("Text similarity (%)")
    ax.set_ylabel("Skill match (%)")
    ax.set_title("Text similarity vs skill match (dashed lines = medians)")
    corner = dict(transform=ax.transAxes, fontsize=8, color="grey", style="italic")
    ax.text(0.02, 0.97, "high skill match,\nlittle matching text", va="top", **corner)
    ax.text(0.98, 0.97, "strong on both", va="top", ha="right", **corner)
    ax.text(0.98, 0.03, "relevant text,\nfew listed skills", va="bottom", ha="right", **corner)
    ax.text(0.02, 0.03, "weak on both", va="bottom", **corner)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def comparison_chart(subset):
    """Grouped bars comparing the chosen candidates on the three scores."""
    if subset.empty:
        return None
    plot_df = subset.assign(Candidate_label=candidate_labels(subset))
    melted = plot_df.melt(id_vars="Candidate_label",
                          value_vars=["Text Similarity (%)", "Skill Match (%)", "Final Score (%)"],
                          var_name="Score", value_name="Value")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=melted, x="Score", y="Value", hue="Candidate_label", ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylim(0, 112)
    ax.set_xlabel("")
    ax.set_ylabel("Score (%)")
    ax.set_title("Candidate comparison")
    ax.legend(title="Candidate", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def close_figure(fig):
    plt.close(fig)