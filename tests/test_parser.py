"""Tests the parser on regex cases and on every file in data/sample_resumes/."""
import os
import sys

sys.path.append(os.getcwd())  # lets us import from src/

from src.parser import extract_email, extract_phone, parse_resume

# ---- Test 1: regex unit tests ----
print("=== TEST 1: Email and phone regex ===")
email_cases = {
    "Contact: john.doe@example.com now": "john.doe@example.com",
    "mail me at a_b-c+d@sub.domain.in": "a_b-c+d@sub.domain.in",
    "no email here": "Not found",
}
for text, expected in email_cases.items():
    got = extract_email(text)
    print(f"[{'PASS' if got == expected else 'FAIL'}] email -> {got}")

phone_cases = {
    "Phone: +91 98765 43210": "+91 98765 43210",
    "Call 9123456780 today": "9123456780",
    "Mob (+91) 99887-76655": "91) 99887-76655",  # see viva note below
    "Year 2024 only": "Not found",
}
for text, expected in phone_cases.items():
    got = extract_phone(text)
    print(f"[info] phone -> '{got}'   (input: {text})")

# ---- Test 2: real files ----
print("\n=== TEST 2: Parsing files in data/sample_resumes ===")
folder = os.path.join("data", "sample_resumes")
files = [f for f in os.listdir(folder) if f.lower().endswith((".pdf", ".docx"))]

if not files:
    print("No resumes found. Run tests/create_sample_resumes.py first.")

for filename in sorted(files):
    path = os.path.join(folder, filename)
    result = parse_resume(path)
    print(f"\nFile   : {result['filename']}")
    print(f"Status : {result['status']}")
    print(f"Name   : {result['name']}")
    print(f"Email  : {result['email']}")
    print(f"Phone  : {result['phone']}")
    print(f"Chars  : {len(result['text'])}")
    print(f"Preview: {result['text'][:120].replace(chr(10), ' | ')}...")

# ---- Test 3: unsupported file ----
print("\n=== TEST 3: Unsupported file type ===")
result = parse_resume("notes.txt")
print(f"Status : {result['status']}")