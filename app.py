"""ATS Resume Checker - Streamlit + Groq.

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get
an ATS score plus concrete improvements.
"""

import io
import json
import os
import re

import streamlit as st
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# Groq model IDs (see https://console.groq.com/docs/models for the live list).
MODEL_OPTIONS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B (best quality, production)",
    "openai/gpt-oss-20b": "GPT-OSS 20B (fastest, production)",
    "qwen/qwen3.6-27b": "Qwen 3.6 27B (preview)",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"
MAX_RESUME_CHARS = 12000
MAX_JD_CHARS = 6000

# Weights must add up to 1.0
CATEGORY_WEIGHTS = {
    "keyword_match": 0.30,
    "formatting_parseability": 0.20,
    "content_impact": 0.25,
    "structure_sections": 0.15,
    "readability_length": 0.10,
}
CATEGORY_LABELS = {
    "keyword_match": "Keyword match",
    "formatting_parseability": "Formatting & ATS parseability",
    "content_impact": "Content & impact (metrics, action verbs)",
    "structure_sections": "Structure & sections",
    "readability_length": "Readability & length",
}

SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and \
professional resume reviewer. You evaluate resumes strictly and honestly. \
Respond with a single valid JSON object and nothing else."""

USER_PROMPT_TEMPLATE = """Analyze the resume below for ATS compatibility.

{jd_block}

Score each category from 0 to 100 (be realistic; an average resume scores 55-70):
- keyword_match: relevant skills/keywords {keyword_hint}
- formatting_parseability: can an ATS parse it (clean text, standard headings, \
no tables/columns/graphics evidence, consistent dates, contact info present)
- content_impact: quantified achievements, strong action verbs, clear results
- structure_sections: presence and order of Contact, Summary, Experience, \
Education, Skills, etc.
- readability_length: concise, no typos, consistent tense, suitable length

Return JSON with EXACTLY this shape:
{{
  "category_scores": {{
    "keyword_match": <int>,
    "formatting_parseability": <int>,
    "content_impact": <int>,
    "structure_sections": <int>,
    "readability_length": <int>
  }},
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<string>", "..."],
  "issues": ["<specific problem found in this resume>", "..."],
  "improvements": [
    {{"priority": "high|medium|low", "suggestion": "<specific, actionable fix>"}}
  ],
  "missing_keywords": ["<keyword>", "..."],
  "rewrite_examples": [
    {{"original": "<weak line from the resume>", "improved": "<stronger rewrite>"}}
  ]
}}

Rules: 3-6 strengths, 4-8 issues, 5-10 improvements, up to 12 missing keywords, \
2-4 rewrite examples taken from the actual resume. Never invent experience the \
candidate does not have.

RESUME:
\"\"\"
{resume}
\"\"\"
"""


# --------------------------------------------------------------------------
# File reading
# --------------------------------------------------------------------------
def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from a PDF, DOCX or TXT upload."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# LLM response handling
# --------------------------------------------------------------------------
def parse_json_response(raw: str) -> dict:
    """Parse JSON from a model reply, tolerating code fences / <think> blocks / chatter."""
    if not raw:
        raise ValueError("The model returned an empty response.")
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("Could not read the model's response as JSON.")


def _clamp(value, low=0, high=100) -> int:
    try:
        return max(low, min(high, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _str_list(value, limit):
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out[:limit]


def normalise_result(data: dict) -> dict:
    """Validate the model output and compute the overall ATS score ourselves."""
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from the model.")

    raw_scores = data.get("category_scores")
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    scores = {key: _clamp(raw_scores.get(key, 0)) for key in CATEGORY_WEIGHTS}
    overall = _clamp(sum(scores[k] * w for k, w in CATEGORY_WEIGHTS.items()))

    improvements = []
    for item in data.get("improvements", []) if isinstance(data.get("improvements"), list) else []:
        if isinstance(item, dict) and str(item.get("suggestion", "")).strip():
            priority = str(item.get("priority", "medium")).lower().strip()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            improvements.append({"priority": priority, "suggestion": str(item["suggestion"]).strip()})
        elif isinstance(item, str) and item.strip():
            improvements.append({"priority": "medium", "suggestion": item.strip()})
    order = {"high": 0, "medium": 1, "low": 2}
    improvements.sort(key=lambda x: order[x["priority"]])

    rewrites = []
    for item in data.get("rewrite_examples", []) if isinstance(data.get("rewrite_examples"), list) else []:
        if isinstance(item, dict) and item.get("original") and item.get("improved"):
            rewrites.append({"original": str(item["original"]).strip(), "improved": str(item["improved"]).strip()})

    return {
        "overall": overall,
        "category_scores": scores,
        "summary": str(data.get("summary", "")).strip(),
        "strengths": _str_list(data.get("strengths"), 8),
        "issues": _str_list(data.get("issues"), 10),
        "improvements": improvements[:12],
        "missing_keywords": _str_list(data.get("missing_keywords"), 15),
        "rewrite_examples": rewrites[:5],
    }


def build_prompt(resume_text: str, job_description: str) -> str:
    jd = job_description.strip()
    if jd:
        jd_block = f'TARGET JOB DESCRIPTION:\n"""\n{jd[:MAX_JD_CHARS]}\n"""'
        hint = "that appear in the job description (and which of them are missing)"
    else:
        jd_block = "No job description was provided; judge against general best practice for the candidate's apparent field."
        hint = "typical for the candidate's apparent field and seniority"
    return USER_PROMPT_TEMPLATE.format(jd_block=jd_block, keyword_hint=hint, resume=resume_text[:MAX_RESUME_CHARS])


def analyze_resume(client, model: str, resume_text: str, job_description: str = "") -> dict:
    """Send the resume to Groq and return the validated analysis."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(resume_text, job_description)},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8000,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Some models reject response_format; retry once without it.
        if "response_format" in str(exc).lower() or "json" in str(exc).lower():
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0.2, max_tokens=8000
            )
        else:
            raise
    content = response.choices[0].message.content
    return normalise_result(parse_json_response(content))


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def score_band(score: int):
    if score >= 80:
        return "Excellent", "green"
    if score >= 65:
        return "Good", "orange"
    if score >= 50:
        return "Needs work", "orange"
    return "Poor", "red"


def get_api_key(sidebar_key: str) -> str:
    if sidebar_key.strip():
        return sidebar_key.strip()
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")


def render_results(result: dict):
    label, color = score_band(result["overall"])
    st.subheader("Your ATS score")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall ATS score", f"{result['overall']} / 100")
        st.markdown(f"**Rating:** :{color}[{label}]")
    with col2:
        if result["summary"]:
            st.write(result["summary"])
    st.progress(result["overall"] / 100)

    st.markdown("#### Score breakdown")
    for key, label_text in CATEGORY_LABELS.items():
        val = result["category_scores"][key]
        st.write(f"{label_text}: **{val}/100**")
        st.progress(val / 100)

    if result["strengths"]:
        st.markdown("#### What's working")
        for item in result["strengths"]:
            st.markdown(f"- ✅ {item}")

    if result["issues"]:
        st.markdown("#### Problems found")
        for item in result["issues"]:
            st.markdown(f"- ⚠️ {item}")

    if result["improvements"]:
        st.markdown("#### Improvements to make")
        icons = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}
        for item in result["improvements"]:
            st.markdown(f"- **{icons[item['priority']]}:** {item['suggestion']}")

    if result["missing_keywords"]:
        st.markdown("#### Keywords to consider adding")
        st.write(", ".join(f"`{k}`" for k in result["missing_keywords"]))

    if result["rewrite_examples"]:
        st.markdown("#### Example rewrites")
        for ex in result["rewrite_examples"]:
            st.markdown(f"**Before:** {ex['original']}")
            st.markdown(f"**After:** {ex['improved']}")
            st.divider()

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(result, indent=2),
        file_name="ats_report.json",
        mime="application/json",
    )


def main():
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Checker")
    st.caption("Upload your resume to get an ATS score and specific ways to improve it.")

    with st.sidebar:
        st.header("Settings")
        sidebar_key = st.text_input("Groq API key", type="password", help="Get a free key at console.groq.com")
        model = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda m: MODEL_OPTIONS[m],
            index=list(MODEL_OPTIONS.keys()).index(DEFAULT_MODEL),
        )
        st.info("Scores are AI estimates. Real ATS software varies, so use this as guidance.")

    uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area(
        "Job description (optional, but gives a much better keyword analysis)",
        height=180,
        placeholder="Paste the job posting here...",
    )

    if st.button("Analyze resume", type="primary"):
        if uploaded is None:
            st.warning("Please upload a resume first.")
            return
        api_key = get_api_key(sidebar_key)
        if not api_key:
            st.error("Add your Groq API key in the sidebar (or set GROQ_API_KEY in Streamlit secrets).")
            return

        try:
            text = clean_text(extract_text(uploaded.name, uploaded.getvalue()))
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        if len(text) < 100:
            st.error(
                "Almost no text could be extracted. If your resume is a scanned image, "
                "an ATS would not be able to read it either. Export a text-based PDF or DOCX."
            )
            return
        if len(text) > MAX_RESUME_CHARS:
            st.info("Your resume is long; only the first part was analysed.")

        try:
            with st.spinner("Analysing your resume..."):
                client = Groq(api_key=api_key)
                result = analyze_resume(client, model, text, job_description)
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "invalid_api_key" in msg.lower() or "authentication" in msg.lower():
                st.error("Invalid Groq API key. Please check it and try again.")
            elif "429" in msg or "rate limit" in msg.lower() or "rate_limit" in msg.lower():
                st.error("Groq rate limit reached. Wait a moment, or try a different model.")
            elif "model" in msg.lower() and ("not found" in msg.lower() or "decommissioned" in msg.lower()):
                st.error("That model is not available on Groq right now. Pick another one in the sidebar.")
            else:
                st.error(f"Analysis failed: {msg}")
            return

        render_results(result)


if __name__ == "__main__":
    main()
