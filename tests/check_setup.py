"""Verifies that all required libraries are installed correctly."""
import importlib

libraries = {
    "pandas": "pandas",
    "numpy": "numpy",
    "pdfplumber": "pdfplumber",
    "docx": "python-docx",
    "nltk": "nltk",
    "sklearn": "scikit-learn",
    "streamlit": "streamlit",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}

all_ok = True
for module, package in libraries.items():
    try:
        lib = importlib.import_module(module)
        version = getattr(lib, "__version__", "installed")
        print(f"[OK]   {package:15} {version}")
    except ImportError:
        print(f"[FAIL] {package:15} NOT INSTALLED")
        all_ok = False

# NLTK data needed for text preprocessing
import nltk
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    nltk.download(resource, quiet=True)
print("[OK]   NLTK data downloaded")

print("\nSETUP COMPLETE" if all_ok else "\nSOME LIBRARIES FAILED - share the output with me")