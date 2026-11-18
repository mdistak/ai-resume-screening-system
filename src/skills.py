"""
skills.py
Dictionary-based skill extraction, JD skill splitting and skill matching.
The skill knowledge base lives in data/skills_db.json (easy to extend).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from src.preprocessing import clean_text

SKILLS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "skills_db.json"

# Lines like "Good to have:" start the optional-skills block of a job description
OPTIONAL_MARKER_RE = re.compile(
    r"^[ \t]*(?:good to have|nice to have|preferred(?: skills?)?|optional(?: skills?)?|bonus)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


@lru_cache(maxsize=1)
def load_skills_db() -> dict:
    with open(SKILLS_DB_PATH, encoding="utf-8") as file:
        return json.load(file)


def _compile_pattern(terms: list):
    """Build one regex for all aliases of a skill, with word-boundary protection."""
    ordered = sorted({t.lower() for t in terms}, key=len, reverse=True)
    parts = [r"\s+".join(re.escape(word) for word in term.split()) for term in ordered]
    body = "|".join(parts)
    return re.compile(rf"(?<![a-z0-9+#])(?:{body})(?![a-z0-9+#])")


@lru_cache(maxsize=1)
def _patterns() -> dict:
    return {skill: _compile_pattern(info["match"])
            for skill, info in load_skills_db().items()}


def get_all_skill_names() -> list:
    return sorted(load_skills_db().keys(), key=str.lower)


def extract_skills(text: str) -> list:
    """Return the sorted list of known skills found in the text."""
    cleaned = clean_text(text)
    found = [skill for skill, rx in _patterns().items() if rx.search(cleaned)]
    return sorted(found, key=str.lower)


def split_jd_sections(jd_text: str):
    """Split a job description into (required_text, optional_text)."""
    marker = OPTIONAL_MARKER_RE.search(jd_text)
    if not marker:
        return jd_text, ""
    rest = jd_text[marker.end():]
    gap = re.search(r"\n\s*\n", rest)          # optional block ends at first blank line
    end = marker.end() + (gap.start() if gap else len(rest))
    optional_text = jd_text[marker.start():end]
    required_text = jd_text[:marker.start()] + "\n" + jd_text[end:]
    return required_text, optional_text


def extract_jd_skills(jd_text: str) -> dict:
    """Return {'required': [...], 'optional': [...]} from a job description."""
    required_text, optional_text = split_jd_sections(jd_text)
    required = extract_skills(required_text)
    optional = [s for s in extract_skills(optional_text) if s not in required]
    return {"required": required, "optional": optional}


def compare_skills(candidate_skills: list, required_skills: list) -> dict:
    """Skill Match (%) = matched required skills / total required skills x 100."""
    candidate = set(candidate_skills)
    matched = [s for s in required_skills if s in candidate]
    missing = [s for s in required_skills if s not in candidate]
    total = len(required_skills)
    percent = round(len(matched) / total * 100, 2) if total else 0.0
    return {"matched": matched, "missing": missing,
            "match_percent": percent, "total_required": total}


def group_by_category(skills: list) -> dict:
    """Group skills by category (useful for charts and future features)."""
    db = load_skills_db()
    grouped = {}
    for skill in skills:
        grouped.setdefault(db[skill]["category"], []).append(skill)
    return grouped