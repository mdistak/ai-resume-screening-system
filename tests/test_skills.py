"""Tests skill extraction, JD splitting and skill matching."""
import os
import sys

sys.path.append(os.getcwd())

from src.parser import parse_resume
from src.skills import (compare_skills, extract_jd_skills, extract_skills,
                        group_by_category, load_skills_db, split_jd_sections)


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")


print(f"Skills in database: {len(load_skills_db())}\n")

print("=== TEST 1: Boundary and alias tests ===")
s = extract_skills("Worked with Java and JavaScript")
check("Java and JavaScript both found", {"Java", "JavaScript"} <= set(s))
s = extract_skills("Strong in JavaScript only")
check("Java NOT found inside JavaScript", "Java" not in s)
s = extract_skills("Used MySQL and NoSQL databases")
check("MySQL and NoSQL found", {"MySQL", "NoSQL"} <= set(s))
check("SQL NOT found inside MySQL/NoSQL", "SQL" not in s)
s = extract_skills("C++ and C# developer")
check("C++ and C# found", {"C++", "C#"} <= set(s))
check("plain C not wrongly detected", "C" not in s)
s = extract_skills("I know C programming and R language")
check("C and R detected with context words", {"C", "R"} <= set(s))
check("random letter r is NOT a skill", "R" not in extract_skills("see section r of the report"))
check("sklearn alias -> Scikit-learn", "Scikit-learn" in extract_skills("used sklearn and scikit learn"))
check("long form -> NLP", "NLP" in extract_skills("Natural Language Processing project"))
check("PowerBI alias -> Power BI", "Power BI" in extract_skills("dashboards in PowerBI"))
check("excellent is NOT Excel", "Excel" not in extract_skills("excellent communication"))
s = extract_skills("Hosted code on GitHub")
check("GitHub found, Git not", "GitHub" in s and "Git" not in s)
check("Node.js found", "Node.js" in extract_skills("Backend with Node.js and Express"))
check("TF-IDF found", "TF-IDF" in extract_skills("Vectorized text using TF-IDF."))

print("\n=== TEST 2: JD splitting ===")
sample_jd = "Required: Python, SQL\n\nGood to have: Tableau\n\nNotes: Docker is used in our team"
req_text, opt_text = split_jd_sections(sample_jd)
result = extract_jd_skills(sample_jd)
print("Result:", result)
check("required = Python, SQL, Docker", set(result["required"]) == {"Python", "SQL", "Docker"})
check("optional = Tableau", result["optional"] == ["Tableau"])

print("\n=== TEST 3: Skill comparison ===")
r = compare_skills(["Python", "Pandas"], ["Python", "SQL", "Pandas", "Git"])
print(r)
check("matched = Python, Pandas", r["matched"] == ["Python", "Pandas"])
check("missing = SQL, Git", r["missing"] == ["SQL", "Git"])
check("percent = 50.0", r["match_percent"] == 50.0)
check("empty required list gives 0.0", compare_skills(["Python"], [])["match_percent"] == 0.0)

print("\n=== TEST 4: Real resumes vs sample job description ===")
jd_path = os.path.join("data", "sample_jd", "data_scientist_jd.txt")
with open(jd_path, encoding="utf-8") as f:
    jd = extract_jd_skills(f.read())
print("JD required :", jd["required"])
print("JD optional :", jd["optional"])

folder = os.path.join("data", "sample_resumes")
for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith((".pdf", ".docx")):
        continue
    text = parse_resume(os.path.join(folder, filename))["text"]
    skills = extract_skills(text)
    result = compare_skills(skills, jd["required"])
    print(f"\n{filename}")
    print("  All skills found :", skills)
    print("  By category      :", group_by_category(skills))
    print("  Matched          :", result["matched"])
    print("  Missing          :", result["missing"])
    print(f"  Skill match      : {result['match_percent']}%  "
          f"({len(result['matched'])}/{result['total_required']})")