"""Tests for step 2: skill evidence flag, blind screening and score explanation."""
import os
import sys

import pandas as pd

sys.path.append(os.getcwd())

from src.anonymize import anonymize_resumes, reveal_identities
from src.matcher import compute_text_similarity, explain_text_similarity, rank_candidates
from src.parser import parse_resume
from src.preprocessing import preprocess_text
from src.skills import review_note, skill_evidence


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


with open(os.path.join("data", "sample_jd", "data_scientist_jd.txt"), encoding="utf-8") as f:
    jd_text = f.read()
folder = os.path.join("data", "sample_resumes")
resumes = [parse_resume(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
           if f.lower().endswith((".pdf", ".docx"))]

print("=== TEST 1: Skill evidence (unit tests) ===")
skill_list = "Skills\nPython, Pandas, NumPy, SQL, Git, Docker, AWS, Spark, Tableau, Excel"
e = skill_evidence(skill_list)
check("bare list of 10 skills: evidence 0", len(e["found"]) == 10 and e["evidence_ratio"] == 0)
check("bare list is flagged", "10 of 10" in review_note(e))

sentences = ("Built a churn model using Python and Pandas. "
             "Cleaned data with SQL and visualised results in Tableau.")
e = skill_evidence(sentences)
check("skills inside sentences: evidence 100%", e["evidence_ratio"] == 1.0)
check("described skills are not flagged", review_note(e) == "")

mixed = skill_list + "\nBuilt a pipeline using Python, Pandas, SQL, Spark and AWS on large datasets."
e = skill_evidence(mixed)
check("list + description: evidence 50%", abs(e["evidence_ratio"] - 0.5) < 1e-9)
check("half supported is not flagged", review_note(e) == "")
check("short skill line (2 skills) never flagged", review_note(skill_evidence("Python, SQL")) == "")
check("empty text -> no crash, no flag", review_note(skill_evidence("")) == "")

print("\n=== TEST 2: Skill evidence on the sample resumes ===")
flags = {}
for r in resumes:
    ev = skill_evidence(r["text"])
    flags[r["filename"]] = review_note(ev)
    print(f"  {r['filename']:32} skills={len(ev['found']):2}  evidence={ev['evidence_ratio'] * 100:5.1f}%"
          f"  flagged={'YES' if flags[r['filename']] else 'no'}")
check("keyword-only resume (Karan) is flagged", flags["Karan_Mehta.docx"] != "")
check("Rahul (skills + described work) is not flagged", flags["Rahul_Sharma.docx"] == "")
check("Sneha (skills + described work) is not flagged", flags["Sneha_Iyer.docx"] == "")

print("\n=== TEST 3: Blind screening ===")
anon, mapping = anonymize_resumes(resumes)
check("names replaced", all(a["name"].startswith("Candidate ") for a in anon))
check("files replaced", all(a["filename"].startswith("Resume ") for a in anon))
check("emails and phones hidden", all(a["email"] == "Hidden" and a["phone"] == "Hidden" for a in anon))
check("one identity-map entry per resume", len(mapping) == len(resumes))
for original, a in zip(resumes, anon):
    if original["name"] != "Name not detected":
        assert original["name"].lower() not in a["text"].lower(), original["name"]
    if original["email"] != "Not found":
        assert original["email"].lower() not in a["text"].lower(), original["email"]
check("names and emails removed from the text", True)

normal_df = rank_candidates(resumes, jd_text)
blind_df = rank_candidates(anon, jd_text)
leaked = [r["name"] for r in resumes if r["name"] != "Name not detected" and
          blind_df.astype(str).apply(lambda c: c.str.contains(r["name"], case=False, regex=False)).any().any()]
check("no real name visible in the blind results", not leaked)

revealed = reveal_identities(blind_df, mapping).set_index("File")
normal = normal_df.set_index("File")
check("same files after reveal", set(revealed.index) == set(normal.index))
check("names restored after reveal",
      all(revealed.loc[f, "Candidate"] == normal.loc[f, "Candidate"] for f in normal.index))
check("skill match unchanged by blinding",
      all(revealed.loc[f, "Skill Match (%)"] == normal.loc[f, "Skill Match (%)"] for f in normal.index))
shift = (revealed["Text Similarity (%)"] - normal["Text Similarity (%)"]).abs().max()
print(f"[info] largest text-similarity change caused by removing names: {shift:.2f} points")

lone = {"filename": "scan_01.pdf", "name": "Name not detected", "email": "Not found",
        "phone": "Not found", "text": "python sql", "status": "OK"}
a, m = anonymize_resumes([lone])
check("unknown name falls back to the file name after reveal",
      a[0]["name"] == "Candidate 01" and m["Resume 01"]["name"] == "scan_01")

print("\n=== TEST 4: Score explanation ===")
explanations = explain_text_similarity(jd_text, resumes, top_n=None)
sims = compute_text_similarity(preprocess_text(jd_text),
                               [preprocess_text(r["text"]) for r in resumes]) * 100
check("one explanation per resume", len(explanations) == len(resumes))
check("all term shares add up to the text similarity",
      all(abs(sum(p for _, p in terms) - s) < 1e-6 for terms, s in zip(explanations, sims)))
check("terms sorted from largest share",
      all([p for _, p in t] == sorted([p for _, p in t], reverse=True) for t in explanations))
check("blank JD -> empty explanations, no crash",
      all(t == [] for t in explain_text_similarity("   ", resumes)))
check("empty resume -> empty explanation",
      explain_text_similarity(jd_text, [dict(lone, text="")]) == [[]])
rahul = [t for r, t in zip(resumes, explain_text_similarity(jd_text, resumes)) if r["filename"].startswith("Rahul")][0]
print("[info] Rahul, top terms:", ", ".join(f"{t} (+{p:.2f})" for t, p in rahul[:6]))