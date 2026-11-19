"""Unit tests for evaluation metrics + edge cases of the matching engine."""
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

from src.evaluation import precision_recall_f1_at_k, rank_correlation
from src.matcher import rank_candidates
from src.parser import parse_resume


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def fake(filename, text):
    return {"filename": filename, "name": filename.split(".")[0], "email": "",
            "phone": "", "text": text, "status": "OK"}


with open(os.path.join("data", "sample_jd", "data_scientist_jd.txt"), encoding="utf-8") as f:
    jd_text = f.read()

folder = os.path.join("data", "sample_resumes")
real = [parse_resume(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
        if f.lower().endswith((".pdf", ".docx"))]

print("=== TEST 1: Precision / Recall / F1 (hand-calculated) ===")
ranked, relevant = ["a", "b", "c", "d"], {"a", "c"}
m = precision_recall_f1_at_k(ranked, relevant, 2)
check("k=2 -> P=0.5, R=0.5, F1=0.5", (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5))
m = precision_recall_f1_at_k(ranked, relevant, 1)
check("k=1 -> P=1.0, R=0.5, F1=0.6667", (m["precision"], m["recall"], m["f1"]) == (1.0, 0.5, 0.6667))
m = precision_recall_f1_at_k(ranked, relevant, 4)
check("k=4 -> P=0.5, R=1.0, F1=0.6667", (m["precision"], m["recall"], m["f1"]) == (0.5, 1.0, 0.6667))
m = precision_recall_f1_at_k(ranked, set(), 2)
check("no relevant candidates -> zeros, no crash", m == {"precision": 0.0, "recall": 0.0, "f1": 0.0})

print("\n=== TEST 2: Rank correlation ===")
table = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "z": [4, 3, 2, 1]})
corr = rank_correlation(table, "x")
check("identical ranking -> 1.0", corr["y"] == 1.0)
check("reversed ranking -> -1.0", corr["z"] == -1.0)

print("\n=== TEST 3: Edge cases ===")
text = "python pandas sql machine learning projects"
df = rank_candidates([fake("a.txt", text), fake("b.txt", text)], jd_text)
check("duplicate resumes get equal scores",
      df.loc[0, "Final Score (%)"] == df.loc[1, "Final Score (%)"])

df = rank_candidates([fake("only.txt", text)], jd_text)
check("single resume works, score within 0-100", len(df) == 1 and 0 <= df.loc[0, "Final Score (%)"] <= 100)

no_skill_jd = "We need a hardworking person with good communication and teamwork"
df = rank_candidates(real, no_skill_jd)
check("JD with no known skills -> skill match 0 for everyone, no crash",
      (df["Skill Match (%)"] == 0).all())
print("       (the dashboard must warn the recruiter in this case)")

df = rank_candidates(real, "   ")
check("blank JD -> all scores 0, no crash", (df["Final Score (%)"] == 0).all())

bad = parse_resume("notes.txt")
df = rank_candidates(real + [bad], jd_text)
check("unreadable file ranks last with score 0",
      df.iloc[-1]["File"] == "notes.txt" and df.iloc[-1]["Final Score (%)"] == 0.0)

print("\n=== TEST 4: Batch dependence of TF-IDF (known limitation) ===")
rahul = [r for r in real if r["filename"].startswith("Rahul")][0]
alone = rank_candidates([rahul], jd_text).loc[0, "Text Similarity (%)"]
in_batch = rank_candidates(real, jd_text).query("File == @rahul['filename']")["Text Similarity (%)"].iloc[0]
print(f"[info] Rahul text similarity alone      : {alone}")
print(f"[info] Rahul text similarity in a batch : {in_batch}")
print("[info] The numbers differ because IDF is computed on the batch. Ranking, not the absolute value, matters.")