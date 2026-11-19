"""Tests TF-IDF, cosine similarity, final score and ranking."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from src.matcher import (build_vectorizer, calculate_final_score, cosine_manual,
                         compute_text_similarity, rank_candidates)
from src.parser import parse_resume
from src.preprocessing import preprocess_text

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 60)


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


print("=== TEST 1: TF-IDF worked example (3 tiny documents) ===")
docs = ["python sql machine learning",            # pretend job description
        "python machine learning pandas",         # similar resume
        "java spring web development"]            # unrelated resume
vec = build_vectorizer(ngram_range=(1, 1))
matrix = vec.fit_transform(docs)
table = pd.DataFrame(matrix.toarray().round(3), columns=vec.get_feature_names_out(),
                     index=["JD", "Resume_A", "Resume_B"])
print(table)

sk = compute_text_similarity(docs[0], docs[1:], ngram_range=(1, 1))
manual = [cosine_manual(matrix[0].toarray(), matrix[i].toarray()) for i in (1, 2)]
print("sklearn cosine :", sk.round(4))
print("manual cosine  :", np.round(manual, 4))
check("manual formula == sklearn", np.allclose(sk, manual))
check("similar resume scores higher than unrelated", sk[0] > sk[1])
check("unrelated resume scores 0", sk[1] == 0.0)
check("identical text scores 1.0", np.isclose(
    compute_text_similarity("python sql", ["python sql"], (1, 1))[0], 1.0))

print("\n=== TEST 2: Tokenizer keeps technical terms ===")
v = build_vectorizer(ngram_range=(1, 1))
v.fit(["c++ scikit-learn node.js c r"])
vocab = set(v.get_feature_names_out())
check("c++ kept", "c++" in vocab)
check("scikit-learn kept", "scikit-learn" in vocab)
check("node.js kept", "node.js" in vocab)
check("single letters c and r kept", {"c", "r"} <= vocab)

print("\n=== TEST 3: Final score formula ===")
check("0.7*80 + 0.3*50 = 71.0", calculate_final_score(80, 50) == 71.0)
check("100 and 100 -> 100.0", calculate_final_score(100, 100) == 100.0)
check("custom weights 50/50", calculate_final_score(80, 40, 0.5, 0.5) == 60.0)
try:
    calculate_final_score(80, 50, 0.9, 0.3)
    check("bad weights rejected", False)
except ValueError:
    check("bad weights rejected", True)

print("\n=== TEST 4: Ranking sanity with fake resumes ===")
jd_path = os.path.join("data", "sample_jd", "data_scientist_jd.txt")
with open(jd_path, encoding="utf-8") as f:
    jd_text = f.read()

def fake(filename, text):
    return {"filename": filename, "name": filename.split(".")[0], "email": "",
            "phone": "", "text": text, "status": "OK"}

fakes = [fake("unrelated.txt", "Java Spring Boot React HTML CSS web developer building websites"),
         fake("copy_of_jd.txt", jd_text),
         fake("empty.txt", "")]
df = rank_candidates(fakes, jd_text)
print(df[["Rank", "Candidate", "Text Similarity (%)", "Skill Match (%)", "Final Score (%)"]])
check("resume identical to JD ranks first", df.loc[0, "Candidate"] == "copy_of_jd")
check("empty resume ranks last with 0 score", df.loc[2, "Candidate"] == "empty"
      and df.loc[2, "Final Score (%)"] == 0.0)
check("no results for empty input list", rank_candidates([], jd_text).empty)

print("\n=== TEST 5: Real sample resumes ===")
folder = os.path.join("data", "sample_resumes")
resumes = [parse_resume(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
           if f.lower().endswith((".pdf", ".docx"))]
result = rank_candidates(resumes, jd_text)
print(result[["Rank", "Candidate", "Text Similarity (%)", "Skill Match (%)", "Final Score (%)"]])
print()
for _, row in result.iterrows():
    print(f"#{row['Rank']} {row['Candidate']}")
    print(f"   Matched: {row['Matched Skills'] or '-'}")
    print(f"   Missing: {row['Missing Skills'] or '-'}")