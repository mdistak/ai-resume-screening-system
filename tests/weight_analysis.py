"""Shows how the ranking changes when the 70/30 weights are changed."""
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

from src.evaluation import rank_correlation, weight_sensitivity
from src.parser import parse_resume

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

with open(os.path.join("data", "sample_jd", "data_scientist_jd.txt"), encoding="utf-8") as f:
    jd_text = f.read()

folder = os.path.join("data", "sample_resumes")
resumes = [parse_resume(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
           if f.lower().endswith((".pdf", ".docx"))]
print(f"Resumes analysed: {len(resumes)}\n")

ranks, scores = weight_sensitivity(resumes, jd_text)
print("RANK of each candidate (columns = text weight / skill weight)")
print(ranks.sort_values("70/30"))
print("\nFINAL SCORE of each candidate")
print(scores.loc[ranks.sort_values("70/30").index])

print("\nSpearman correlation with the default 70/30 ranking (1.0 = same order):")
for label, value in rank_correlation(ranks, "70/30").items():
    print(f"  {label:>7} : {value}")

moved = ranks[ranks.nunique(axis=1) > 1]
print(f"\nCandidates whose rank changes with the weights: {len(moved)} of {len(ranks)}")
print(moved.sort_values("70/30") if len(moved) else "None, ranking is stable.")