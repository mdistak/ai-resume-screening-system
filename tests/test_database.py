"""Tests the SQLite history layer on a temporary database."""
import os
import sqlite3
import sys
import tempfile
from contextlib import closing

import numpy as np

sys.path.append(os.getcwd())

from src.database import (clear_history, delete_run, get_run_details, get_run_results,
                          init_db, list_runs, make_job_title, save_analysis)
from src.matcher import rank_candidates
from src.parser import parse_resume
from src.skills import extract_jd_skills


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


def table_names(db):
    with closing(sqlite3.connect(db)) as c:
        return {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


with open(os.path.join("data", "sample_jd", "data_scientist_jd.txt"), encoding="utf-8") as f:
    jd = f.read()
required = extract_jd_skills(jd)["required"]
folder = os.path.join("data", "sample_resumes")
resumes = [parse_resume(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
           if f.lower().endswith((".pdf", ".docx"))]
df = rank_candidates(resumes, jd, required)

with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    db = os.path.join(tmp, "test_history.db")

    print("=== TEST 1: Tables ===")
    init_db(db)
    check("both tables created", {"analysis_runs", "candidate_results"} <= table_names(db))

    print("\n=== TEST 2: Save and read back ===")
    run1 = save_analysis(df, jd, 0.7, 0.3, required, db)
    check("first run_id is 1", run1 == 1)
    runs = list_runs(db)
    check("one run listed", len(runs) == 1)
    check("candidate_count matches", runs.loc[0, "candidate_count"] == len(df))
    check("top candidate matches", runs.loc[0, "top_candidate"] == df.loc[0, "Candidate"])
    check("top score matches", runs.loc[0, "top_score"] == df.loc[0, "Final Score (%)"])

    saved = get_run_results(run1, db)
    check("same number of rows", len(saved) == len(df))
    check("ranks are 1..n in order", list(saved["Rank"]) == list(range(1, len(df) + 1)))
    check("files in the same order", list(saved["File"]) == list(df["File"]))
    check("final scores identical", np.allclose(saved["Final Score (%)"], df["Final Score (%)"]))
    check("matched skills identical", list(saved["Matched Skills"]) == list(df["Matched Skills"]))

    details = get_run_details(run1, db)
    check("job title stored", details["job_title"] == make_job_title(jd))
    check("weights stored", (details["text_weight"], details["skill_weight"]) == (0.7, 0.3))
    check("required skills stored", details["required_skills"] == ", ".join(required))
    check("missing run returns None", get_run_details(999, db) is None)

    print("\n=== TEST 3: Privacy (no contact details or resume text stored) ===")
    with closing(sqlite3.connect(db)) as c:
        columns = {row[1] for row in c.execute("PRAGMA table_info(candidate_results)")}
        columns |= {row[1] for row in c.execute("PRAGMA table_info(analysis_runs)")}
    check("no email/phone/resume text columns",
          not ({"email", "phone", "text", "resume_text"} & columns))

    print("\n=== TEST 4: Several runs, delete, cascade ===")
    run2 = save_analysis(df.head(3), jd, 0.5, 0.5, required, db)
    runs = list_runs(db)
    check("two runs, newest first", len(runs) == 2 and runs.loc[0, "run_id"] == run2)
    delete_run(run1, db)
    check("run 1 deleted", list(list_runs(db)["run_id"]) == [run2])
    check("its candidate rows deleted too (CASCADE)", len(get_run_results(run1, db)) == 0)
    check("run 2 rows untouched", len(get_run_results(run2, db)) == 3)

    print("\n=== TEST 5: SQL injection attempt is stored as plain text ===")
    nasty = "Robert'); DROP TABLE analysis_runs;--\nmore text"
    run3 = save_analysis(df.head(1), nasty, 0.7, 0.3, [], db)
    check("tables still exist", {"analysis_runs", "candidate_results"} <= table_names(db))
    check("text saved safely", get_run_details(run3, db)["job_title"] == nasty.splitlines()[0])

    print("\n=== TEST 6: Clear history and helpers ===")
    clear_history(db)
    check("no runs left", list_runs(db).empty)
    with closing(sqlite3.connect(db)) as c:
        left = c.execute("SELECT COUNT(*) FROM candidate_results").fetchone()[0]
    check("no candidate rows left", left == 0)
    check("title skips blank lines", make_job_title("\n\n  Data Analyst \nrest") == "Data Analyst")
    check("empty text gets a default title", make_job_title("  ") == "Untitled job description")