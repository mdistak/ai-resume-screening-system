"""Computes Precision/Recall/F1@k from HUMAN labels (data/evaluation/labels.csv)."""
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

from src.evaluation import evaluate_ranking
from src.matcher import rank_candidates
from src.parser import parse_resume

RESUME_DIR = os.path.join("data", "sample_resumes")
JD_PATH = os.path.join("data", "sample_jd", "data_scientist_jd.txt")
LABELS_PATH = os.path.join("data", "evaluation", "labels.csv")
TEMPLATE_PATH = os.path.join("data", "evaluation", "labels_template.csv")

files = sorted(f for f in os.listdir(RESUME_DIR) if f.lower().endswith((".pdf", ".docx")))


def create_template_and_exit():
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    pd.DataFrame({"filename": files, "suitable": ""}).to_csv(TEMPLATE_PATH, index=False)
    print(f"A template was created: {TEMPLATE_PATH}")
    print("\nWhat to do:")
    print("  1. Open the template and read each resume against the job description.")
    print("  2. In column 'suitable' write 1 (suitable for the job) or 0 (not suitable).")
    print("     Decide BEFORE looking at the system's scores, so the labels stay unbiased.")
    print("  3. Rename it to labels.csv and run this script again:")
    print("     Rename-Item data\\evaluation\\labels_template.csv labels.csv")
    sys.exit(0)


if not os.path.exists(LABELS_PATH) or os.path.getsize(LABELS_PATH) == 0:
    if os.path.exists(LABELS_PATH):
        print("labels.csv exists but is EMPTY. Delete it and use the template instead:")
        print("  Remove-Item data\\evaluation\\labels.csv\n")
    create_template_and_exit()

try:
    labels_df = pd.read_csv(LABELS_PATH, encoding="utf-8-sig")
except pd.errors.EmptyDataError:
    print("labels.csv has no readable content. Delete it: Remove-Item data\\evaluation\\labels.csv")
    sys.exit(1)

labels_df.columns = [c.strip().lower() for c in labels_df.columns]
if not {"filename", "suitable"} <= set(labels_df.columns):
    print("Error: labels.csv must have the columns 'filename' and 'suitable'.")
    sys.exit(1)

labels_df = labels_df.dropna(subset=["suitable"])
if labels_df.empty:
    print("No labels filled yet. Write 1 or 0 in the 'suitable' column and save the file.")
    sys.exit(1)

try:
    labels_df["suitable"] = labels_df["suitable"].astype(int)
except ValueError:
    print("Error: column 'suitable' must contain only the numbers 0 or 1.")
    sys.exit(1)
if not set(labels_df["suitable"]) <= {0, 1}:
    print("Error: column 'suitable' must contain only 0 or 1.")
    sys.exit(1)

labels = dict(zip(labels_df["filename"].str.strip(), labels_df["suitable"]))

unknown = [f for f in labels if f not in files]
if unknown:
    print(f"Warning: these labelled files were not found in {RESUME_DIR}: {unknown}")
if len(labels) < 2:
    print("Label at least 2 resumes.")
    sys.exit(1)
if len(set(labels.values())) < 2:
    print("Warning: all labels are the same (all 0 or all 1). Metrics will not be meaningful.")

with open(JD_PATH, encoding="utf-8") as f:
    jd_text = f.read()

resumes = [parse_resume(os.path.join(RESUME_DIR, f)) for f in files if f in labels]
ranked = rank_candidates(resumes, jd_text)
ranked["Human Label"] = ranked["File"].map(labels)

print(ranked[["Rank", "Candidate", "Final Score (%)", "Human Label"]].to_string(index=False))
print("\nPrecision / Recall / F1 at top-k:")
print(evaluate_ranking(ranked, labels).to_string(index=False))
print(f"\nNOTE: computed on {len(labels)} human-labelled resumes only. "
      "This is not a general accuracy of the system.")