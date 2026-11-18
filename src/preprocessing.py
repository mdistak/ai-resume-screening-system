"""
preprocessing.py
Cleans raw resume / job-description text for TF-IDF and skill matching.

Two outputs:
  clean_text()      -> lowercase readable text (used for skill phrase matching)
  preprocess_text() -> tokenized, stopword-free, lemmatized text (used for TF-IDF)
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data():
    """Download NLTK resources if missing (also needed on Streamlit Cloud)."""
    for path, name in [("corpora/stopwords", "stopwords"),
                       ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b[a-z0-9-]+\.(?:com|org|io|in)\b\S*)")
PHONE_RE = re.compile(r"\(?\+?\d[\d\s\-().]{8,16}\d")
# Keeps C++, C#, Node.js, scikit-learn; first char must be a letter
TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.\-]*")

# Words that appear in every resume header and carry no matching value
RESUME_NOISE = {"email", "phone", "mobile", "linkedin", "github", "address",
                "resume", "curriculum", "vitae", "contact"}

# Tech terms the lemmatizer would damage (e.g. pandas -> panda, css -> cs)
PROTECTED_TERMS = {
    "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn", "plotly",
    "tensorflow", "pytorch", "keras", "streamlit", "spacy", "nltk",
    "css", "js", "aws", "sas", "gis", "ios", "sql", "mysql", "postgresql",
    "nlp", "analytics", "hadoop", "kubernetes", "docker", "django", "flask",
    "linux", "git", "github", "excel", "tableau", "kafka", "redis",
}

STOP_WORDS = set(stopwords.words("english")) | RESUME_NOISE
_lemmatizer = WordNetLemmatizer()


def _remove_phone_numbers(text: str) -> str:
    """Replace only real phone numbers (10-13 digits), not year ranges."""
    def replacer(match):
        digits = re.sub(r"\D", "", match.group(0))
        return " " if 10 <= len(digits) <= 13 else match.group(0)
    return PHONE_RE.sub(replacer, text)


def clean_text(text: str) -> str:
    """Lowercase, remove emails/URLs/phones/symbols, collapse whitespace."""
    text = text.lower()
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = _remove_phone_numbers(text)
    # keep letters, digits and the symbols used in tech terms: + # . - /
    text = re.sub(r"[^a-z0-9+#.\-/\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list:
    """Split cleaned text into tokens, keeping tech terms like c++ and node.js."""
    tokens = []
    for token in TOKEN_RE.findall(text):
        token = token.strip(".-")          # remove trailing sentence dots
        if re.fullmatch(r"x{4,}", token):   # placeholder like xxxxxxxxxx
             continue
        if len(token) >= 2 or token in ("c", "r"):   # keep C and R languages
            tokens.append(token)
    return tokens


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize(tokens: list) -> list:
    """Reduce words to base form (noun pass, then verb pass)."""
    result = []
    for token in tokens:
        if token in PROTECTED_TERMS or not token.isalpha():
            result.append(token)           # leave tech terms like c++, node.js alone
        else:
            base = _lemmatizer.lemmatize(token, pos="n")
            base = _lemmatizer.lemmatize(base, pos="v")
            result.append(base)
    return result


def preprocess_tokens(text: str) -> list:
    """Full pipeline returning a list of tokens."""
    return lemmatize(remove_stopwords(tokenize(clean_text(text))))


def preprocess_text(text: str) -> str:
    """Full pipeline returning one string (the input format TF-IDF needs)."""
    return " ".join(preprocess_tokens(text))


def pipeline_steps(text: str) -> dict:
    """Returns the output of every stage. Useful for demo and viva."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(no_stop)
    return {"1_cleaned": cleaned, "2_tokens": tokens,
            "3_no_stopwords": no_stop, "4_lemmatized": lemmas}