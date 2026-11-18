"""
parser.py
Extracts raw text and contact details from PDF and DOCX resumes.
Works with file paths (testing) and file-like objects (Streamlit uploads).
"""
import os
import re

import pdfplumber
from docx import Document

SUPPORTED_EXTENSIONS = (".pdf", ".docx")

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Candidate phone strings: digits with optional +, spaces, hyphens, brackets, dots
PHONE_CANDIDATE_PATTERN = re.compile(r"\+?\d[\d\s\-().]{8,16}\d")


def extract_text_from_pdf(file) -> str:
    """Extract text from every page of a PDF. `file` = path or file-like object."""
    pages_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
    return "\n".join(pages_text)


def extract_text_from_docx(file) -> str:
    """Extract text from paragraphs and tables of a DOCX file."""
    document = Document(file)
    parts = [p.text for p in document.paragraphs if p.text.strip()]

    # Many resumes keep skills/experience inside tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)
    return "\n".join(parts)


def extract_text(file, filename: str = None) -> str:
    """Pick the right extractor based on the file extension."""
    if filename is None:
        filename = getattr(file, "name", str(file))
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file)
    if extension == ".docx":
        return extract_text_from_docx(file)
    raise ValueError(f"Unsupported file type: '{extension}'. Use PDF or DOCX.")


def extract_email(text: str) -> str:
    """Return the first email address found, or 'Not found'."""
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else "Not found"


def extract_phone(text: str) -> str:
    """Return the first valid-looking phone number (10 to 13 digits)."""
    for candidate in PHONE_CANDIDATE_PATTERN.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 13:
            return candidate.strip()
    return "Not found"


def extract_name(text: str) -> str:
    """
    Heuristic: the candidate name is usually one of the first few lines,
    2-4 alphabetic words, without '@' or digits. Not always correct.
    """
    for line in text.splitlines()[:6]:
        line = line.strip()
        words = line.split()
        if (2 <= len(words) <= 4
                and all(w.replace(".", "").isalpha() for w in words)):
            return line.title()
    return "Name not detected"


def parse_resume(file, filename: str = None) -> dict:
    """Full pipeline for one resume. Never crashes: errors are returned in the dict."""
    if filename is None:
        filename = getattr(file, "name", str(file))
    filename = os.path.basename(filename)

    result = {
        "filename": filename,
        "name": "",
        "email": "",
        "phone": "",
        "text": "",
        "status": "OK",
    }
    try:
        text = extract_text(file, filename)
        if len(text.strip()) < 30:
            result["status"] = "Warning: very little text (scanned/image PDF?)"
        result["text"] = text
        result["name"] = extract_name(text)
        result["email"] = extract_email(text)
        result["phone"] = extract_phone(text)
    except Exception as error:
        result["status"] = f"Error: {error}"
    return result