"""
database.py
SQLite storage for analysis history (standard library only, no extra install).

Privacy by design: only scores, skills and names are stored.
Resume text, email addresses and phone numbers are NOT saved.
"""
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    job_title       TEXT    NOT NULL,
    jd_text         TEXT    NOT NULL,
    text_weight     REAL    NOT NULL,
    skill_weight    REAL    NOT NULL,
    required_skills TEXT    NOT NULL,
    candidate_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
    rank_no         INTEGER NOT NULL,
    candidate_name  TEXT    NOT NULL,
    file_name       TEXT    NOT NULL,
    text_similarity REAL    NOT NULL,
    skill_match     REAL    NOT NULL,
    final_score     REAL    NOT NULL,
    matched_skills  TEXT,
    missing_skills  TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON candidate_results(run_id);
"""


def _connect(db_path=None):
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")   # needed for ON DELETE CASCADE
    return connection


def init_db(db_path=None):
    """Create the tables if they do not exist yet."""
    with closing(_connect(db_path)) as connection:
        connection.executescript(SCHEMA)


def make_job_title(jd_text: str, limit: int = 80) -> str:
    """First non-empty line of the job description, used as its title."""
    for line in jd_text.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return "Untitled job description"


def save_analysis(df, jd_text, text_weight, skill_weight, required_skills, db_path=None) -> int:
    """
    Save one ranking (the DataFrame from rank_candidates) and return its run_id.
    Both inserts happen in ONE transaction: either everything is saved or nothing.
    All values are passed as parameters (?), never pasted into the SQL text.
    """
    init_db(db_path)
    with closing(_connect(db_path)) as connection:
        with connection:                       # commit on success, rollback on error
            cursor = connection.execute(
                "INSERT INTO analysis_runs (created_at, job_title, jd_text, text_weight, "
                "skill_weight, required_skills, candidate_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), make_job_title(jd_text),
                 jd_text, float(text_weight), float(skill_weight),
                 ", ".join(required_skills), int(len(df))))
            run_id = cursor.lastrowid
            rows = [(run_id, int(r["Rank"]), str(r["Candidate"]), str(r["File"]),
                     float(r["Text Similarity (%)"]), float(r["Skill Match (%)"]),
                     float(r["Final Score (%)"]), str(r["Matched Skills"]),
                     str(r["Missing Skills"])) for _, r in df.iterrows()]
            connection.executemany(
                "INSERT INTO candidate_results (run_id, rank_no, candidate_name, file_name, "
                "text_similarity, skill_match, final_score, matched_skills, missing_skills) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return run_id


def list_runs(db_path=None, limit: int = 50) -> pd.DataFrame:
    """Newest runs first, with the top candidate of each run."""
    init_db(db_path)
    sql = """
        SELECT r.run_id, r.created_at, r.job_title, r.candidate_count,
               r.text_weight, r.skill_weight,
               (SELECT candidate_name FROM candidate_results c
                 WHERE c.run_id = r.run_id ORDER BY c.rank_no LIMIT 1) AS top_candidate,
               (SELECT final_score FROM candidate_results c
                 WHERE c.run_id = r.run_id ORDER BY c.rank_no LIMIT 1) AS top_score
        FROM analysis_runs r
        ORDER BY r.run_id DESC
        LIMIT ?
    """
    with closing(_connect(db_path)) as connection:
        return pd.read_sql_query(sql, connection, params=(limit,))


def get_run_details(run_id: int, db_path=None):
    """Settings of one run as a dict, or None if it does not exist."""
    init_db(db_path)
    with closing(_connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM analysis_runs WHERE run_id = ?",
                                 (int(run_id),)).fetchone()
    return dict(row) if row else None


def get_run_results(run_id: int, db_path=None) -> pd.DataFrame:
    """Candidate table of one run, using the same column names as the dashboard."""
    init_db(db_path)
    sql = """
        SELECT rank_no AS "Rank", candidate_name AS "Candidate", file_name AS "File",
               text_similarity AS "Text Similarity (%)", skill_match AS "Skill Match (%)",
               final_score AS "Final Score (%)", matched_skills AS "Matched Skills",
               missing_skills AS "Missing Skills"
        FROM candidate_results WHERE run_id = ? ORDER BY rank_no
    """
    with closing(_connect(db_path)) as connection:
        return pd.read_sql_query(sql, connection, params=(int(run_id),))


def delete_run(run_id: int, db_path=None):
    """Delete one run; its candidate rows are removed automatically (CASCADE)."""
    init_db(db_path)
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("DELETE FROM analysis_runs WHERE run_id = ?", (int(run_id),))


def clear_history(db_path=None):
    """Delete every saved run."""
    init_db(db_path)
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("DELETE FROM analysis_runs")