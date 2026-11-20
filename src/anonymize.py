"""
anonymize.py
Blind screening: hide identity before scoring, reveal it afterwards.
Hidden: name, email, phone and file name. Names, emails and phones are also removed
from the resume text, so they cannot influence the score.
This does NOT remove every possible proxy (college, city, etc.).
"""
import re

PLACEHOLDERS = ("", "Not found", "Name not detected")


def anonymize_resumes(resumes: list):
    """Returns (anonymous_resumes, identity_map). identity_map: 'Resume 01' -> real details."""
    anonymous, identity_map = [], {}
    for index, resume in enumerate(resumes, start=1):
        file_label = f"Resume {index:02d}"
        text = resume["text"]
        for value in (resume["name"], resume["email"], resume["phone"]):
            if value not in PLACEHOLDERS:
                text = re.sub(re.escape(value), " ", text, flags=re.IGNORECASE)

        real_name = resume["name"]
        if real_name in PLACEHOLDERS:
            real_name = resume["filename"].rsplit(".", 1)[0]
        identity_map[file_label] = {"name": real_name, "file": resume["filename"],
                                    "email": resume["email"], "phone": resume["phone"]}
        anonymous.append(dict(resume, name=f"Candidate {index:02d}", filename=file_label,
                              email="Hidden", phone="Hidden", text=text))
    return anonymous, identity_map


def reveal_identities(df, identity_map: dict):
    """Put the real name, file, email and phone back into a ranking DataFrame."""
    revealed = df.copy()
    info = [identity_map.get(f) for f in revealed["File"]]
    revealed["Candidate"] = [i["name"] if i else c for i, c in zip(info, revealed["Candidate"])]
    revealed["Email"] = [i["email"] if i else e for i, e in zip(info, revealed["Email"])]
    revealed["Phone"] = [i["phone"] if i else p for i, p in zip(info, revealed["Phone"])]
    revealed["File"] = [i["file"] if i else f for i, f in zip(info, revealed["File"])]
    return revealed