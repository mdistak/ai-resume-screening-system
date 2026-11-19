"""Tests the chart functions and saves PNG files into screenshots/."""
import os
import sys

sys.path.append(os.getcwd())

from matplotlib.figure import Figure

from src.matcher import rank_candidates
from src.parser import parse_resume
from src.skills import extract_jd_skills
from src.visualization import (close_figure, comparison_chart, save_figure,
                               score_breakdown_chart, skill_heatmap, skill_matrix,
                               text_vs_skill_chart)


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


with open(os.path.join("data", "sample_jd", "data_scientist_jd.txt"), encoding="utf-8") as f:
    jd_text = f.read()
required = extract_jd_skills(jd_text)["required"]

folder = os.path.join("data", "sample_resumes")
resumes = [parse_resume(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
           if f.lower().endswith((".pdf", ".docx"))]
df = rank_candidates(resumes, jd_text)
print(f"{len(df)} candidates, {len(required)} required skills\n")

print("=== TEST 1: skill matrix ===")
matrix = skill_matrix(df, required)
check("shape = candidates x required skills", matrix.shape == (len(df), len(required)))
check("only 0 and 1 in the matrix", set(matrix.to_numpy().ravel()) <= {0, 1})
rahul_row = [i for i in matrix.index if "Rahul" in i][0]
priya_row = [i for i in matrix.index if "Priya" in i][0]
check("Rahul has all 9 required skills", matrix.loc[rahul_row].sum() == 9)
check("Priya has only 1 required skill", matrix.loc[priya_row].sum() == 1)
check("top_n limits rows", skill_matrix(df.head(5), required).shape[0] == 5)

print("\n=== TEST 2: charts are created and saved ===")
charts = {
    "chart_score_breakdown.png": score_breakdown_chart(df, 0.70, 0.30),
    "chart_skill_heatmap.png": skill_heatmap(df, required),
    "chart_text_vs_skill.png": text_vs_skill_chart(df),
    "chart_comparison.png": comparison_chart(df.head(3)),
}
for filename, fig in charts.items():
    path = os.path.join("screenshots", filename)
    check(f"{filename} is a Figure", isinstance(fig, Figure))
    save_figure(fig, path)
    check(f"{filename} saved ({os.path.getsize(path) // 1024} KB)", os.path.getsize(path) > 5000)
    close_figure(fig)

print("\n=== TEST 3: edge cases ===")
check("heatmap returns None without required skills", skill_heatmap(df, []) is None)
check("comparison returns None for empty selection", comparison_chart(df.head(0)) is None)
single = rank_candidates(resumes[:1], jd_text)
for name, fig in [("score breakdown", score_breakdown_chart(single, 0.7, 0.3)),
                  ("heatmap", skill_heatmap(single, required)),
                  ("scatter", text_vs_skill_chart(single)),
                  ("comparison", comparison_chart(single))]:
    check(f"{name} works with a single candidate", isinstance(fig, Figure))
    close_figure(fig)