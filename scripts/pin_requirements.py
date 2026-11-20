"""Writes requirements.txt with the exact versions installed in this environment."""
from importlib.metadata import version

PACKAGES = ["pandas", "numpy", "pdfplumber", "python-docx", "nltk",
            "scikit-learn", "streamlit", "matplotlib", "seaborn"]

lines = [f"{name}=={version(name)}" for name in PACKAGES]
with open("requirements.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nrequirements.txt updated (direct dependencies only).")