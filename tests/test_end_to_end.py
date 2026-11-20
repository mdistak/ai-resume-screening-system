"""End-to-end and robustness tests that follow the same path as the Streamlit app."""
import io
import os
import random
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document

sys.path.append(os.getcwd())

from src.anonymize import anonymize_resumes, reveal_identities
from src.database import list_runs, save_analysis
from src.matcher import rank_candidates
from src.parser import parse_resume
from src.skills import extract_jd_skills, extract_skills


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def docx_bytes(paragraphs):
    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def blank_pdf_bytes():
    figure = plt.figure(figsize=(3, 3))
    axes = figure.add_axes([0, 0, 1, 1])
    axes.axis("off")
    axes.plot([0, 1], [0, 1])               # a drawing without any text
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf")
    plt.close(figure)
    return buffer.getvalue()


with open(os.path.join("data", "sample_jd", "data_scientist_jd.txt"), encoding="utf-8") as f:
    jd_text = f.read()

print("=== TEST 1: Bad and unusual files (parsed from bytes, like uploads) ===")
bad_pdf = parse_resume(io.BytesIO(b"this is not a pdf"), filename="bad.pdf")
check("corrupted PDF -> error status, no crash", bad_pdf["status"].startswith("Error"))
bad_docx = parse_resume(io.BytesIO(b"this is not a docx"), filename="bad.docx")
check("corrupted DOCX -> error status, no crash", bad_docx["status"].startswith("Error"))
blank_docx = parse_resume(io.BytesIO(docx_bytes([])), filename="blank.docx")
check("blank DOCX -> warning status", blank_docx["status"].startswith("Warning"))
blank_pdf = parse_resume(io.BytesIO(blank_pdf_bytes()), filename="blank.pdf")
check("PDF without text -> not reported as OK", blank_pdf["status"] != "OK")
txt = parse_resume(io.BytesIO(b"hello"), filename="notes.txt")
check("unsupported type -> error status", txt["status"].startswith("Error"))

hindi = parse_resume(io.BytesIO(docx_bytes(
    ["\u0928\u092e\u0938\u094d\u0924\u0947, \u092e\u0948\u0902 \u0921\u0947\u091f\u093e "
     "\u0938\u093e\u0907\u0902\u091f\u093f\u0938\u094d\u091f \u0939\u0942\u0901\u0964",
     "Skills: Python, SQL, Pandas"])), filename="hindi.docx")
check("Hindi text does not crash and English skills are still found",
      hindi["status"] == "OK" and {"Python", "SQL", "Pandas"} <= set(extract_skills(hindi["text"])))

big_text = ["Built machine learning models in Python and SQL for data analysis. " * 15] * 120
start = time.time()
big = parse_resume(io.BytesIO(docx_bytes(big_text)), filename="big.docx")
rank_candidates([big], jd_text)
elapsed = time.time() - start
print(f"[info] very long resume ({len(big['text'].split())} words) parsed and ranked in {elapsed:.1f} s")
check("very long resume finishes in under 30 s", elapsed < 30)

print("\n=== TEST 2: Normal pipeline on the sample resumes ===")
folder = os.path.join("data", "sample_resumes")
resumes = []
for name in sorted(os.listdir(folder)):
    if name.lower().endswith((".pdf", ".docx")):
        with open(os.path.join(folder, name), "rb") as handle:
            resumes.append(parse_resume(io.BytesIO(handle.read()), filename=name))
df = rank_candidates(resumes, jd_text)
print(f"[info] {len(df)} resumes ranked")
check("one row per resume", len(df) == len(resumes))
check("ranks are 1..n", list(df["Rank"]) == list(range(1, len(df) + 1)))
check("scores within 0-100", df[["Text Similarity (%)", "Skill Match (%)", "Final Score (%)"]]
      .apply(lambda c: c.between(0, 100).all()).all())
check("sorted by final score, highest first", df["Final Score (%)"].is_monotonic_decreasing)
check("deterministic: same input gives identical output", rank_candidates(resumes, jd_text).equals(df))

shuffled = resumes[:]
random.Random(42).shuffle(shuffled)
diff = (df.set_index("File")["Final Score (%)"]
        - rank_candidates(shuffled, jd_text).set_index("File")["Final Score (%)"]).abs().max()
check(f"upload order does not change scores (max diff {diff:.3f})", diff <= 0.011)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
back = pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")
check("CSV keeps all columns", list(back.columns) == list(df.columns))
check("CSV keeps the scores", (back["Final Score (%)"] - df["Final Score (%)"]).abs().max() < 1e-9)

mixed = rank_candidates([bad_pdf, blank_docx] + resumes[:3], jd_text)
check("bad and blank files do not break ranking",
      len(mixed) == 5 and set(mixed.tail(2)["File"]) == {"bad.pdf", "blank.docx"})

print("\n=== TEST 3: Job description variants ===")
original = extract_jd_skills(jd_text)
check("lower-case JD gives the same skills", extract_jd_skills(jd_text.lower()) == original)
check("Windows line endings (CRLF) give the same skills",
      extract_jd_skills(jd_text.replace("\n", "\r\n")) == original)
start = time.time()
rank_candidates(resumes[:3], jd_text * 200)
check("very long JD (200x) finishes in under 30 s", time.time() - start < 30)
unicode_df = rank_candidates(resumes[:3], "\u0921\u0947\u091f\u093e \u0938\u093e\u0907\u0902\u091f\u093f\u0938\u094d\u091f Python SQL")
check("JD with Hindi words does not crash", len(unicode_df) == 3)

print("\n=== TEST 4: Blind screening and history together ===")
anon, mapping = anonymize_resumes(resumes)
blind_df = rank_candidates(anon, jd_text)
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    db = os.path.join(tmp, "e2e.db")
    save_analysis(blind_df, jd_text, 0.7, 0.3, original["required"], db)
    runs = list_runs(db)
    check("history of a blind run stores anonymous names", runs.loc[0, "top_candidate"].startswith("Candidate "))
check("reveal restores real file names", set(reveal_identities(blind_df, mapping)["File"]) == set(df["File"]))

print("\n=== TEST 5: 60 resumes at once ===")
many = [dict(r, filename=f"{i}_{r['filename']}") for i in range(5) for r in resumes]
start = time.time()
big_df = rank_candidates(many, jd_text)
elapsed = time.time() - start
print(f"[info] {len(many)} resumes ranked in {elapsed:.1f} s")
check("60 resumes ranked in under 60 s", len(big_df) == len(many) and elapsed < 60)