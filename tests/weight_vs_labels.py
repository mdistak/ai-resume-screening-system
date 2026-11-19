"""Which weight pair agrees best with the human labels? (exploratory, small sample)"""
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

from src.evaluation import DEFAULT_WEIGHT_PAIRS, precision_recall_f1_at_k
from src.matcher import rank_candidates
from src.parser import parse_resume

RESUME_DIR = os.path.join("data", "sample_resumes")
JD_PATH = os.path.join("data", "sample_jd", "data_scientist_jd.txt")
LABELS_PATH = os.path.join("data", "evaluation", "labels.csv")

labels_df = pd.read_csv(LABELS_PATH, encoding="utf-8-sig")
labels_df.columns = [c.strip().lower() for c in labels_df.columns]
labels_df = labels_df.dropna(subset=["suitable"])
labels = dict(zip(labels_df["filename"].str.strip(), labels_df["suitable"].astype(int)))
relevant = {f for f, v in labels.items() if v == 1}
k = len(relevant)

resumes = [parse_resume(os.path.join(RESUME_DIR, f))
           for f in sorted(os.listdir(RESUME_DIR)) if f in labels]
with open(JD_PATH, encoding="utf-8") as f:
    jd_text = f.read()

rows = []
for text_w, skill_w in DEFAULT_WEIGHT_PAIRS:
    ranked = rank_candidates(resumes, jd_text, None, text_w, skill_w)
    metrics = precision_recall_f1_at_k(list(ranked["File"]), relevant, k)
    rows.append({"weights (text/skill)": f"{round(text_w * 100)}/{round(skill_w * 100)}",
                 f"Precision@{k}": metrics["precision"],
                 f"Top-{k} shortlist": ", ".join(ranked["Candidate"].head(k))})

print(f"{len(labels)} labelled resumes, {k} marked suitable (k = {k})\n")
print(pd.DataFrame(rows).to_string(index=False))
print("\nNOTE: exploratory result on a small synthetic sample with one labeller. "
      "If several weights give the same value, the difference is not meaningful.")