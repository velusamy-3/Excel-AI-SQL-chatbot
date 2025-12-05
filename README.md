# Excel → AI SQL Chatbot (Streamlit)

A Streamlit app that loads an Excel/CSV, generates SQL using an LLM (Mistral), runs safe queries against the uploaded table, and presents friendly human answers.

## Files
- `app.py` — main Streamlit application.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — optional Streamlit config.
- `.gitignore` — recommended.

## Local setup & run
1. Create a virtual environment and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS / Linux
   # venv\Scripts\activate      # Windows PowerShell
   pip install -r requirements.txt
