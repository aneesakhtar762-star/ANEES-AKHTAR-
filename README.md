# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and gives specific improvements, powered by the Groq API.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optionally paste a **job description** for targeted keyword analysis
- Overall **ATS score (0-100)** plus a breakdown: keyword match, formatting, content impact, structure, readability
- Strengths, problems found, prioritised improvements, missing keywords and before/after rewrite examples
- Download the report as JSON

## Models (Groq)

Selectable in the sidebar. Check the [live Groq model list](https://console.groq.com/docs/models) since availability changes.

| Model ID | Notes |
| --- | --- |
| `openai/gpt-oss-120b` | Default, best quality |
| `openai/gpt-oss-20b` | Fastest |
| `qwen/qwen3.6-27b` | Preview model |

To add or change models, edit `MODEL_OPTIONS` at the top of `app.py`.

## Run locally

1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Get a free API key at [console.groq.com](https://console.groq.com).
4. Provide the key in one of three ways:
   - Paste it in the app sidebar, or
   - Set an environment variable: `export GROQ_API_KEY=your_key` (Windows: `set GROQ_API_KEY=your_key`), or
   - Create `.streamlit/secrets.toml` containing `GROQ_API_KEY = "your_key"`
5. Start the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key"
   ```
5. Click **Deploy**.

> Never commit your API key to GitHub. Use Streamlit secrets or the sidebar field.

## Notes and limitations

- Scores are AI estimates, not output from a real ATS. Treat them as guidance.
- Scanned or image-only PDFs have no extractable text; export a text-based PDF or DOCX instead.
- Very long resumes are truncated (first ~12,000 characters) before analysis.
- The overall score is computed in code from the category scores using fixed weights (keywords 30%, content 25%, formatting 20%, structure 15%, readability 10%), so results are consistent in format.

## Project structure

```
app.py             # Streamlit app
requirements.txt   # Python dependencies
README.md          # This file
```
