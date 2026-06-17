#!/usr/bin/env python3
"""
Team Apex — Candidate Ranking Pipeline
Redrob Intelligent Candidate Discovery Challenge

Usage:
    python run_pipeline.py --candidates candidates.jsonl --jd job_description.txt --out team_apex.csv
    python run_pipeline.py --candidates sample_candidates.json --jd job_description.txt --sample-mode
"""

import argparse
import csv
import gzip
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


TODAY = date.today()
CURRENT_YEAR = TODAY.year


# Earliest year each technology existed publicly.
# A candidate claiming 8 years of PyTorch experience is a red flag.
TECH_BIRTH_YEARS = {
    "pytorch": 2016, "transformers": 2017, "bert": 2018,
    "gpt": 2018, "hugging face": 2016, "huggingface": 2016,
    "rag": 2020, "retrieval augmented generation": 2020,
    "langchain": 2022, "llama": 2023, "mistral": 2023,
    "pinecone": 2019, "weaviate": 2018, "qdrant": 2021,
    "faiss": 2017, "sentence-transformers": 2019,
    "stable diffusion": 2022, "chatgpt": 2022,
    "openai api": 2020, "lora": 2021, "qlora": 2023,
    "kubernetes": 2014, "docker": 2013, "react": 2013,
    "flutter": 2018, "swift": 2014, "kotlin": 2016,
}

# Current titles that disqualify a candidate regardless of listed skills.
NON_ENGINEER_TITLES = {
    "marketing", "sales", "hr ", "human resource", "operations manager",
    "operations director", "account manager", "account executive",
    "business development", "content writer", "seo", "brand manager",
    "graphic design", "recruiter", "talent acquisition", "finance",
    "accountant", "project coordinator", "office manager",
    "customer success", "customer support", "support agent",
    "supply chain", "procurement", "legal", "lawyer",
    "civil engineer", "mechanical engineer", "electrical engineer",
}

# Role titles we won't highlight in reasoning.
NON_TECH_ROLE_TITLES = {
    "civil", "mechanical", "electrical", "chemical", "marketing",
    "sales", "hr", "recruiter", "finance", "accountant", "legal",
    "operations", "project manager", "scrum", "business analyst",
    "content", "seo", "brand", "graphic", "designer",
}

INDIA_CITIES = {
    "noida", "pune", "bangalore", "bengaluru", "mumbai", "delhi",
    "hyderabad", "chennai", "kolkata", "gurgaon", "gurugram",
    "ahmedabad", "jaipur", "surat", "lucknow", "kanpur", "nagpur",
    "indore", "thane", "bhopal", "visakhapatnam", "patna", "vadodara",
    "ghaziabad", "ludhiana", "agra", "nashik", "faridabad", "meerut",
    "rajkot", "kochi", "coimbatore", "chandigarh", "mysore", "bhubaneswar",
    "greater noida", "ncr", "navi mumbai", "trivandrum", "thiruvananthapuram",
}

# Universal technical vocabulary used to extract meaningful skills from any JD.
# Terms are tiered: Tier 3 = core technical skills, Tier 2 = strong signal,
# Tier 1 = supporting signal.
TECH_VOCAB_TIER3 = {
    # Vector databases and search infrastructure
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "pgvector", "chroma", "vespa", "typesense",
    "redis search", "algolia", "solr",
    # Retrieval and ranking
    "embeddings", "embedding", "semantic search", "vector search",
    "dense retrieval", "sparse retrieval", "hybrid search", "bm25",
    "reranking", "rerank", "information retrieval", "neural search",
    "recommendation systems", "recommendation", "ranking", "retrieval",
    "rag", "retrieval augmented generation", "ann", "hnsw",
    # Backend / infra
    "kafka", "redis", "postgresql", "postgres", "cassandra", "mongodb",
    "mysql", "dynamodb", "bigquery", "snowflake", "databricks",
    "kubernetes", "docker", "terraform", "airflow", "spark",
    # Mobile / frontend
    "react", "react native", "flutter", "swift", "kotlin", "android",
    "ios", "typescript", "javascript", "nextjs", "nodejs",
}

TECH_VOCAB_TIER2 = {
    # ML frameworks and models
    "pytorch", "tensorflow", "hugging face", "huggingface",
    "sentence transformers", "sentence-transformers", "transformers",
    "bert", "gpt", "llm", "large language model", "fine-tuning",
    "fine tuning", "nlp", "natural language processing", "computer vision",
    "langchain", "llamaindex", "openai", "anthropic", "gemini",
    "scikit-learn", "sklearn", "xgboost", "lightgbm",
    # Evaluation
    "ndcg", "mrr", "map", "precision", "recall", "a/b testing",
    "evaluation", "offline evaluation", "online evaluation",
    "f1", "auc", "roc",
    # Languages
    "python", "golang", "java", "scala", "rust", "cpp", "c++",
    "sql", "graphql",
    # Cloud
    "aws", "gcp", "azure", "s3", "ec2", "lambda", "gke", "eks",
    # Practices
    "mlops", "ml pipeline", "feature engineering", "feature store",
    "model serving", "model deployment", "inference", "latency",
    "applied ml", "machine learning", "deep learning",
}

TECH_VOCAB_TIER1 = {
    "production", "shipped", "scale", "scalable", "distributed",
    "microservices", "api", "rest", "grpc", "protobuf",
    "ci/cd", "devops", "monitoring", "observability",
    "data engineering", "etl", "streaming", "batch",
    "neural network", "transformer", "attention", "tokenizer",
}

# Words that carry no signal and should be ignored
GENERIC_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "that", "this", "these", "those", "we", "you", "they", "our",
    "your", "their", "it", "its", "not", "no", "can", "if", "what",
    "how", "who", "which", "when", "where", "why", "also", "more",
    "any", "all", "both", "each", "few", "most", "other", "some",
    "such", "than", "then", "so", "yet", "about", "above", "after",
    "before", "between", "into", "through", "during", "role", "team",
    "work", "build", "use", "using", "strong", "experience", "year",
    "years", "good", "real", "well", "like", "open", "pure", "one",
    "per", "via", "etc", "very", "just", "only", "even", "over",
    "product", "system", "search", "engineer", "shipped", "users",
    "location", "database", "research", "india", "relocation",
    "notice", "period", "days", "preferred", "attitude", "opinions",
    "actually", "built", "means", "think", "write", "find", "painful",
    "style", "abrasive", "stable", "mature", "productive", "unstable",
    "narrow", "explicitly", "genuinely", "active", "clear", "signal",
    "market", "including", "without", "requires", "required", "must",
    "need", "needs", "want", "looking", "ideal", "candidate", "candidates",
    "hiring", "hire", "join", "company", "team", "startup", "series",
    "early", "stage", "founding", "senior", "junior", "lead", "staff",
    "principal", "manager", "director", "head", "vp", "cto",
}


# ─────────────────────────────────────────────────────────────────────────────
# JD PARSER
# ─────────────────────────────────────────────────────────────────────────────

class JobDescription:
    """
    Reads a plain text JD file and extracts everything the pipeline needs:
    experience range, target locations, and skill weights.

    Skill extraction uses a two-pass approach:
    1. Match against a known technical vocabulary (tiered weights)
    2. Catch any remaining technical-looking terms not in the vocab

    This produces precise, signal-rich skill weights instead of generic
    frequency counts that treat "product" and "pinecone" equally.
    """

    def __init__(self, path: str):
        self.path = path
        self.text = Path(path).read_text(encoding="utf-8")
        self.text_lower = self.text.lower()

        self.exp_min, self.exp_max = self._parse_experience()
        self.locations = self._parse_locations()
        self.skill_weights = self._parse_skills()

        print(f"  [jd] {Path(path).name}")
        print(f"  [jd] Experience: {self.exp_min}-{self.exp_max} years")
        print(f"  [jd] Locations: {sorted(self.locations) or 'none specified'}")
        print(f"  [jd] Top skills: {list(self.skill_weights.keys())[:10]}")
        print()

    def _parse_experience(self):
        patterns = [
            r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\+\s*(?:years?|yrs?)',
            r'minimum\s+(\d+)\s*(?:years?|yrs?)',
            r'at least\s+(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, self.text_lower)
            if matches:
                first = matches[0]
                if isinstance(first, tuple):
                    lo, hi = int(first[0]), int(first[1])
                    return max(0, lo - 1), hi + 2
                else:
                    base = int(first)
                    return max(0, base - 1), base + 6
        return 3, 12

    def _parse_locations(self):
        found = set()
        for city in INDIA_CITIES:
            if city in self.text_lower:
                found.add(city)
        if any(c in found for c in ["delhi", "noida", "gurgaon", "gurugram"]):
            found.update(["delhi", "noida", "gurgaon", "gurugram", "ncr", "greater noida"])
        return found

    def _parse_skills(self):
        weights = {}

        # Pass 1: check every known technical term against the JD
        for term in TECH_VOCAB_TIER3:
            if term in self.text_lower:
                weights[term] = 3.0

        for term in TECH_VOCAB_TIER2:
            if term in self.text_lower and term not in weights:
                weights[term] = 2.0

        for term in TECH_VOCAB_TIER1:
            if term in self.text_lower and term not in weights:
                weights[term] = 1.0

        # Pass 2: catch technical terms in the JD not in our vocab
        # Look for hyphenated terms, version strings, and short acronyms
        extra = re.findall(
            r'\b([a-z][a-z0-9]*(?:[-\.][a-z0-9]+)+|[a-z]{2,6}[0-9]+[a-z0-9]*)\b',
            self.text_lower
        )
        for term in extra:
            if term not in weights and term not in GENERIC_WORDS and len(term) >= 2:
                count = self.text_lower.count(term)
                if count > 0:
                    weights[term] = 1.0

        # If JD has very few matches (unusual JD), fall back to frequency counting
        if len(weights) < 5:
            words = re.findall(r'\b[a-z][a-z0-9\-]{2,29}\b', self.text_lower)
            counts = {}
            for w in words:
                if w not in GENERIC_WORDS:
                    counts[w] = self.text_lower.count(w)
            if counts:
                max_c = max(counts.values())
                for k, v in counts.items():
                    if k not in weights:
                        weights[k] = round(1 + (v / max_c) * 0.5, 2)

        return dict(sorted(weights.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple:
    profile = candidate.get("profile", {})
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    yoe     = profile.get("years_of_experience", 0)

    total_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 0 and total_months > yoe * 12 * 1.4 + 6:
        return True, f"Career history ({total_months}mo) exceeds stated experience ({yoe}yr)"

    for skill in skills:
        name    = skill.get("name", "").lower()
        claimed = skill.get("duration_months", 0)
        for tech, birth in TECH_BIRTH_YEARS.items():
            if tech in name:
                ceiling = (CURRENT_YEAR - birth - 1) * 12
                if claimed > ceiling:
                    return True, f"'{skill['name']}' claimed {claimed}mo but only exists since {birth}"

    expert_zero = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    ]
    if len(expert_zero) >= 3:
        return True, f"Expert on {len(expert_zero)} skills with 0 months each"

    for job in career:
        try:
            if int(job.get("start_date", "2000")[:4]) < 1990:
                return True, "Career start date before 1990"
        except (ValueError, IndexError):
            pass

    inflated = [
        s for s in skills
        if s.get("endorsements", 0) > 500 and s.get("duration_months", 12) < 6
    ]
    if inflated:
        return True, "Endorsement count inconsistent with usage duration"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def is_non_engineer(title: str) -> bool:
    t = title.lower()
    return any(token in t for token in NON_ENGINEER_TITLES)


def passes_location(candidate: dict, targets: set) -> bool:
    if candidate.get("redrob_signals", {}).get("willing_to_relocate", False):
        return True
    if not targets:
        return True
    profile = candidate.get("profile", {})
    if profile.get("country", "").lower() not in ("india", "in", ""):
        return False
    loc = profile.get("location", "").lower()
    return any(t in loc for t in targets)


def passes_experience(candidate: dict, exp_min: float, exp_max: float) -> bool:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    return exp_min <= yoe <= exp_max


def experience_fit(yoe: float, exp_min: float, exp_max: float) -> float:
    """
    Returns a multiplier for experience fit. Applied multiplicatively
    on the composite score, not as a small additive component.
    Below the JD range = harsh penalty. Above = moderate penalty.
    """
    sweet_lo = exp_min + 1
    sweet_hi = exp_max - 2
    if sweet_lo <= yoe <= sweet_hi:  return 1.0    # perfect range
    elif exp_min <= yoe < sweet_lo:  return 0.65   # below target — real penalty
    elif sweet_hi < yoe <= exp_max:  return 0.80   # slightly over
    else:                             return 0.40   # way outside


def title_seniority_multiplier(candidate: dict) -> float:
    """
    Penalizes junior titles and rewards senior/staff/lead titles.
    A "Junior ML Engineer" should not outrank a "Staff ML Engineer"
    when skills are similar.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()
    if "junior" in title or "intern" in title:
        return 0.80
    elif any(t in title for t in ["staff", "principal", "distinguished"]):
        return 1.10
    elif any(t in title for t in ["lead", "head", "director", "vp"]):
        return 1.05
    elif "senior" in title:
        return 1.0
    else:
        return 0.95  # mid-level, no title signal


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_multiplier(signals: dict) -> float:
    """
    Scores how hirable a candidate actually is based on platform behaviour.
    Output is 0.2 to 1.2 — applied as a multiplier on the base score.

    A ghost candidate (inactive, unresponsive, long notice) can have a
    perfect resume and still score near zero after this multiplier is applied.
    """
    score   = 0.0
    ceiling = 0.0

    ceiling += 20
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            days = (TODAY - date.fromisoformat(last_active)).days
            if days <= 14:    score += 20
            elif days <= 30:  score += 17
            elif days <= 60:  score += 12
            elif days <= 90:  score += 7
            elif days <= 180: score += 3
        except ValueError:
            pass

    ceiling += 15
    if signals.get("open_to_work_flag", False):
        score += 15

    ceiling += 20
    rrr = signals.get("recruiter_response_rate", 0.0)
    if rrr >= 0.80:    score += 20
    elif rrr >= 0.60:  score += 16
    elif rrr >= 0.40:  score += 11
    elif rrr >= 0.20:  score += 6

    ceiling += 8
    rt = signals.get("avg_response_time_hours", 999)
    if rt <= 4:    score += 8
    elif rt <= 12: score += 6
    elif rt <= 24: score += 4
    elif rt <= 72: score += 2

    ceiling += 12
    notice = signals.get("notice_period_days", 90)
    if notice <= 0:    score += 12
    elif notice <= 15: score += 11
    elif notice <= 30: score += 10
    elif notice <= 60: score += 6
    elif notice <= 90: score += 2

    ceiling += 10
    gh = signals.get("github_activity_score", -1)
    if gh >= 70:   score += 10
    elif gh >= 40: score += 7
    elif gh >= 15: score += 4
    elif gh >= 0:  score += 1

    ceiling += 8
    icr = signals.get("interview_completion_rate", 0.0)
    if icr >= 0.80:   score += 8
    elif icr >= 0.60: score += 5
    elif icr >= 0.40: score += 2

    ceiling += 4
    if signals.get("verified_email", False): score += 2
    if signals.get("verified_phone", False): score += 2

    ceiling += 3
    if signals.get("applications_submitted_30d", 0) > 0: score += 2
    if signals.get("saved_by_recruiters_30d", 0) > 0:    score += 1

    ratio = score / ceiling if ceiling > 0 else 0.0
    return round(min(max(0.2 + ratio, 0.2), 1.2), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _build_combined_weights(jd_weights: dict) -> dict:
    """
    Combine JD-extracted weights with universal tech vocabulary.
    JD terms keep their full weight. Tech vocab terms not mentioned
    in the JD get a reduced but nonzero weight — they're still relevant
    to a technical role even if the JD doesn't list every synonym.

    This is what closes the gap between hardcoded and universal scoring:
    a candidate with PyTorch experience still gets credit for an AI role
    even if the JD only mentions "embeddings" and "vector search".
    """
    combined = dict(jd_weights)
    for term in TECH_VOCAB_TIER3:
        if term not in combined:
            combined[term] = 2.0
    for term in TECH_VOCAB_TIER2:
        if term not in combined:
            combined[term] = 1.5
    for term in TECH_VOCAB_TIER1:
        if term not in combined:
            combined[term] = 0.5
    return combined


_combined_cache = None

def skill_score(candidate: dict, jd_weights: dict) -> float:
    global _combined_cache
    if _combined_cache is None:
        _combined_cache = _build_combined_weights(jd_weights)

    total   = 0.0
    ceiling = 25.0

    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        prof = skill.get("proficiency", "")
        dur  = skill.get("duration_months", 0)

        best = max(
            (w for term, w in _combined_cache.items() if term in name or name in term),
            default=0.0
        )
        if best > 0:
            prof_mult = {
                "beginner": 0.4, "intermediate": 0.7,
                "advanced": 1.0, "expert": 1.2
            }.get(prof, 0.6)
            dur_trust = min(dur / 24.0, 1.0) if dur > 3 else 0.3
            total += best * prof_mult * dur_trust

    return round(min(total / ceiling, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SCORING
# ─────────────────────────────────────────────────────────────────────────────

_model        = None
_jd_embedding = None
_tfidf        = None
_jd_vec       = None


def profile_text(candidate: dict) -> str:
    p     = candidate.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for job in candidate.get("career_history", []):
        parts += [job.get("title", ""), job.get("description", "")]
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(x for x in parts if x)


def load_model(jd_text: str) -> bool:
    global _model, _jd_embedding
    try:
        from sentence_transformers import SentenceTransformer
        print("  [embed] Loading semantic model...", flush=True)
        _model        = SentenceTransformer("all-MiniLM-L6-v2")
        _jd_embedding = _model.encode(jd_text, convert_to_numpy=True)
        print("  [embed] Done.", flush=True)
        return True
    except Exception as e:
        print(f"  [embed] Unavailable ({type(e).__name__}). Using TF-IDF fallback.", flush=True)
        return False


def fit_tfidf(texts: list, jd_text: str):
    global _tfidf, _jd_vec
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(f"  [tfidf] Fitting on {len(texts):,} profiles...", flush=True)
    v       = TfidfVectorizer(ngram_range=(1, 2), max_features=30000,
                              sublinear_tf=True, strip_accents="unicode")
    v.fit([jd_text] + texts)
    _tfidf  = v
    _jd_vec = v.transform([jd_text])
    print("  [tfidf] Done.", flush=True)


def semantic_scores(candidates: list, jd_text: str) -> list:
    texts = [profile_text(c) for c in candidates]

    if _model is not None:
        print(f"  [embed] Scoring {len(texts):,} profiles...", flush=True)
        vecs = _model.encode(texts, batch_size=256,
                             show_progress_bar=True, convert_to_numpy=True)
        return [round(float(max(0.0, 1.0 - cosine_distance(_jd_embedding, v))), 4)
                for v in vecs]

    if _tfidf is None:
        fit_tfidf(texts, jd_text)

    cv  = _tfidf.transform(texts)
    dot = (cv * _jd_vec.T).toarray().flatten()
    cn  = np.array(np.sqrt(cv.multiply(cv).sum(axis=1))).flatten()
    jn  = float(np.sqrt(_jd_vec.multiply(_jd_vec).sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where((cn * jn) > 0, dot / (cn * jn), 0.0)
    mx = sims.max() if sims.max() > 0 else 1.0
    return [round(float(max(0.0, min(s / mx, 1.0))), 4) for s in sims]


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def composite(sem: float, skill: float, beh: float,
               exp: float, title_mult: float) -> float:
    base = 0.45 * sem + 0.55 * skill
    return round(min(max(base * beh * exp * title_mult, 0.0), 1.0), 6)


# ─────────────────────────────────────────────────────────────────────────────
# REASONING
# ─────────────────────────────────────────────────────────────────────────────

def career_is_relevant(job: dict, jd_weights: dict) -> bool:
    if any(t in job.get("title", "").lower() for t in NON_TECH_ROLE_TITLES):
        return False
    desc = job.get("description", "").lower()
    return any(term in desc for term in list(jd_weights.keys())[:50])


def reasoning(candidate: dict, rank: int, skill_sc: float,
              beh: float, jd: "JobDescription") -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])
    career  = candidate.get("career_history", [])

    yoe     = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    loc     = profile.get("location", "")
    notice  = signals.get("notice_period_days", 90)
    rrr     = signals.get("recruiter_response_rate", 0.0)

    prof_order  = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    by_strength = sorted(skills,
        key=lambda s: (prof_order.get(s.get("proficiency", ""), 0),
                       s.get("duration_months", 0)), reverse=True)

    top_skills = []
    for s in by_strength:
        name = s.get("name", "").lower()
        if any(t in name or name in t for t in jd.skill_weights):
            top_skills.append(s["name"])
        if len(top_skills) >= 3:
            break

    parts = []

    if skill_sc >= 0.6:
        match = ", ".join(top_skills[:2]) if top_skills else "strong skill match"
        parts.append(f"{yoe:.0f}yr exp, strong skill match ({match})")
    elif skill_sc >= 0.35:
        hint = f" ({top_skills[0]})" if top_skills else ""
        parts.append(f"{yoe:.0f}yr exp with relevant background{hint}")
    else:
        if yoe < jd.exp_min + 1:
            parts.append(f"{yoe:.0f}yr exp — below JD target range")
        elif yoe > jd.exp_max - 1:
            parts.append(f"{yoe:.0f}yr exp — above JD target range")
        else:
            parts.append(f"{yoe:.0f}yr exp at {company or 'current company'}, limited skill overlap")

    for job in career[:3]:
        if career_is_relevant(job, jd.skill_weights):
            parts.append(
                f"shipped {job.get('title','relevant system')} at "
                f"{job.get('company','prior company')}"
            )
            break

    if beh >= 1.1:
        parts.append(f"active on platform, {rrr:.0%} response rate")
    elif beh < 0.7 and rrr < 0.20:
        parts.append(f"concern: low recruiter response rate ({rrr:.0%})")

    if notice <= 30:
        parts.append(f"available in {notice}d")
    elif notice > 90:
        parts.append(f"long notice period ({notice}d) is a risk")

    if jd.locations and any(t in loc.lower() for t in jd.locations):
        parts.append(f"based in {loc}")
    elif signals.get("willing_to_relocate", False):
        parts.append(f"open to relocating from {loc}")

    if rank <= 10:    label = "Strong fit"
    elif rank <= 30:  label = "Good fit"
    elif rank <= 60:  label = "Moderate fit"
    else:             label = "Partial fit"

    s1 = f"{label}: {'; '.join(parts[:2])}." if parts else f"{label} for role."
    s2 = f"{'; '.join(parts[2:]).capitalize()}." if len(parts) > 2 else ""
    return re.sub(r"\s+", " ", (s1 + " " + s2).strip())[:500]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        def _gz():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return _gz()
    elif p.suffix in (".jsonl", ".ndjson"):
        def _jsonl():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return _jsonl()
    else:
        with open(p, "r", encoding="utf-8") as f:
            return iter(json.load(f))


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────────────────────────────────────

def stage1(candidates_iter, jd: "JobDescription", verbose: bool = True) -> list:
    surviving   = []
    total       = 0
    dropped_loc = dropped_exp = dropped_title = dropped_honey = 0

    for cand in candidates_iter:
        total += 1
        if total % 10000 == 0 and verbose:
            print(f"  [stage1] {total:,} processed | {len(surviving):,} surviving", flush=True)

        if not passes_location(cand, jd.locations):
            dropped_loc += 1; continue

        if not passes_experience(cand, jd.exp_min, jd.exp_max):
            dropped_exp += 1; continue

        if is_non_engineer(cand.get("profile", {}).get("current_title", "")):
            dropped_title += 1; continue

        flagged, reason = is_honeypot(cand)
        if flagged:
            dropped_honey += 1
            if verbose and dropped_honey <= 5:
                print(f"  [honeypot] {cand.get('candidate_id')}: {reason}", flush=True)
            continue

        surviving.append(cand)

    if verbose:
        print(f"\n  [stage1] Processed  : {total:,}")
        print(f"  [stage1] Location   : -{dropped_loc:,}")
        print(f"  [stage1] Experience : -{dropped_exp:,}")
        print(f"  [stage1] Title      : -{dropped_title:,}")
        print(f"  [stage1] Honeypot   : -{dropped_honey:,}")
        print(f"  [stage1] Surviving  : {len(surviving):,}\n")

    return surviving


def stage2(candidates: list, jd: "JobDescription",
           top_n: int = 150, verbose: bool = True) -> list:
    if verbose:
        print(f"  [stage2] Scoring {len(candidates):,} candidates...", flush=True)

    sem_scores = semantic_scores(candidates, jd.text)

    scored = []
    for i, cand in enumerate(candidates):
        sem  = sem_scores[i]
        sk   = skill_score(cand, jd.skill_weights)
        beh  = behavioral_multiplier(cand.get("redrob_signals", {}))
        yoe  = cand.get("profile", {}).get("years_of_experience", 0)
        exp  = experience_fit(yoe, jd.exp_min, jd.exp_max)
        tmult = title_seniority_multiplier(cand)
        comp = composite(sem, sk, beh, exp, tmult)
        scored.append((comp, sk, sem, beh, cand))

    scored.sort(key=lambda x: (-x[0], x[4]["candidate_id"]))
    top = scored[:top_n]

    if verbose:
        print(f"\n  [stage2] Selected {len(top):,} candidates.")
        if top:
            print(f"  [stage2] Score range: {top[-1][0]:.4f} — {top[0][0]:.4f}\n")

    return top


def stage3(top_scored: list, out_path: str,
           jd: "JobDescription", verbose: bool = True) -> list:
    final = top_scored[:100]
    rows  = []

    for rank, (comp, sk, sem, beh, cand) in enumerate(final, start=1):
        rows.append({
            "candidate_id": cand["candidate_id"],
            "rank":         rank,
            "score":        comp,
            "reasoning":    reasoning(cand, rank, sk, beh, jd),
        })

    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = rows[i - 1]["score"]

    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["score"] == rows[i]["score"]:
            j += 1
        if j - i > 1:
            group = sorted(rows[i:j], key=lambda r: r["candidate_id"])
            for k, row in enumerate(group):
                row["rank"] = i + 1 + k
            rows[i:j] = group
        i = j

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print(f"  [stage3] Written {len(rows)} rows to {out_path}\n")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate(csv_path: str) -> bool:
    import importlib.util
    for p in [Path(__file__).parent / "validate_submission.py",
              Path("validate_submission.py")]:
        if p.exists():
            spec   = importlib.util.spec_from_file_location("vs", p)
            mod    = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            errors = mod.validate_submission(csv_path)
            if errors:
                print(f"\n[VALIDATION FAILED] {len(errors)} issue(s):")
                for e in errors:
                    print(f"  {e}")
                return False
            print("[VALIDATION PASSED] Ready to submit.")
            return True
    print("[WARN] validate_submission.py not found.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Team Apex — Candidate Ranking Pipeline")
    parser.add_argument("--candidates",    default="sample_candidates.json")
    parser.add_argument("--jd",           default="job_description.txt")
    parser.add_argument("--out",          default="team_apex.csv")
    parser.add_argument("--top-n-stage2", type=int, default=150)
    parser.add_argument("--quiet",        action="store_true")
    parser.add_argument("--sample-mode",  action="store_true")
    args    = parser.parse_args()
    verbose = not args.quiet

    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  Team Apex — Candidate Ranking Pipeline")
    print(f"  Candidates : {args.candidates}")
    print(f"  JD         : {args.jd}")
    print(f"  Output     : {args.out}")
    print(f"{'='*55}\n")

    if not Path(args.jd).exists():
        print(f"[ERROR] JD file not found: {args.jd}")
        sys.exit(1)

    global _combined_cache
    _combined_cache = None
    print("── Parsing JD ──")
    jd = JobDescription(args.jd)
    load_model(jd.text)

    print("── Stage 1: Filter ──")
    pool = stage1(load_candidates(args.candidates), jd, verbose)

    if len(pool) < 100:
        print(f"[WARN] Only {len(pool)} passed filters. Relaxing location gate...")
        pool = []
        for cand in load_candidates(args.candidates):
            if not passes_experience(cand, jd.exp_min, jd.exp_max): continue
            if is_non_engineer(cand.get("profile", {}).get("current_title", "")): continue
            flagged, _ = is_honeypot(cand)
            if flagged: continue
            pool.append(cand)
        print(f"  After relaxation: {len(pool)}\n")

    print("── Stage 2: Score ──")
    top_n      = min(args.top_n_stage2, max(100, len(pool)))
    top_scored = stage2(pool, jd, top_n=top_n, verbose=verbose)

    if len(top_scored) < 100:
        if args.sample_mode:
            print(f"[INFO] Sample mode — {len(top_scored)} candidates scored.")
        else:
            print("[ERROR] Fewer than 100 candidates. Use the full dataset.")
            sys.exit(1)

    print("── Stage 3: Output ──")
    stage3(top_scored, args.out, jd, verbose)

    print("── Validation ──")
    ok = validate(args.out)

    print(f"\n{'='*55}")
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Output : {args.out}")
    print(f"  Status : {'VALID — ready to submit' if ok else 'INVALID — see errors above'}")
    print(f"{'='*55}\n")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()