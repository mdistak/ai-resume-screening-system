"""Tests the preprocessing pipeline with unit cases and real sample files."""
import os
import sys

sys.path.append(os.getcwd())

from src.parser import parse_resume
from src.preprocessing import clean_text, pipeline_steps, preprocess_tokens


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


print("=== TEST 1: Unit tests ===")
t = preprocess_tokens("Python, Pandas & NumPy! Building dashboards.")
check("tech words kept lowercase", {"python", "pandas", "numpy"} <= set(t))
check("pandas NOT lemmatized to panda", "panda" not in t)
check("dashboards -> dashboard", "dashboard" in t)
check("building -> build", "build" in t)

t = preprocess_tokens("C++ and C# developer using Node.js and scikit-learn")
check("c++ preserved", "c++" in t)
check("c# preserved", "c#" in t)
check("node.js preserved", "node.js" in t)
check("scikit-learn preserved", "scikit-learn" in t)
check("stopwords removed", "and" not in t and "using" in t or "use" in t)

t = preprocess_tokens("Experienced in R programming and SQL")
check("R language kept", "r" in t)

cleaned = clean_text("Mail: john@example.com Ph: +91 98765 43210 www.site.com")
check("email removed", "@" not in cleaned)
check("phone removed", "98765" not in cleaned)
check("url removed", "www" not in cleaned)

cleaned = clean_text("Worked 2021-2024 on projects")
check("year range NOT removed as phone", "2021-2024" in cleaned)

print("\n=== TEST 2: Step by step on a sample sentence ===")
sample = "Built 3 dashboards using Pandas, SQL and Power BI for the sales team."
for step, output in pipeline_steps(sample).items():
    print(f"{step}: {output}")

print("\n=== TEST 3: Real resumes and job description ===")
folder = os.path.join("data", "sample_resumes")
for filename in sorted(os.listdir(folder)):
    if filename.lower().endswith((".pdf", ".docx")):
        text = parse_resume(os.path.join(folder, filename))["text"]
        tokens = preprocess_tokens(text)
        print(f"\n{filename}: {len(text.split())} raw words -> {len(tokens)} tokens")
        print("First 15 tokens:", tokens[:15])

jd_path = os.path.join("data", "sample_jd", "data_scientist_jd.txt")
with open(jd_path, encoding="utf-8") as f:
    jd_tokens = preprocess_tokens(f.read())
print(f"\nJob description -> {len(jd_tokens)} tokens")
print("First 15 tokens:", jd_tokens[:15])