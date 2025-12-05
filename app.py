import os
import re
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from datetime import datetime

# ---------- CONFIG ----------
DB_PATH = "data.db"
TABLE_NAME = "data_table"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-small-latest"


# ---------- DB HELPERS ----------
def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def load_excel_to_db(file) -> pd.DataFrame:
    filename: str = file.name
    df = pd.read_csv(file) if filename.lower().endswith(".csv") else pd.read_excel(file)
    engine = get_engine()
    df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    return df


def existing_data_available() -> bool:
    if not os.path.exists(DB_PATH):
        return False

    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT 1 FROM {TABLE_NAME} LIMIT 1"))
        return True
    except Exception:
        return False


def get_rich_schema_description() -> str:
    engine = get_engine()
    with engine.connect() as conn:
        df_sample = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 50", conn)

    lines = []
    for col, dtype in zip(df_sample.columns, df_sample.dtypes):
        if dtype == "object":
            with engine.connect() as conn:
                distinct_vals = pd.read_sql(
                    text(f'SELECT DISTINCT "{col}" AS val FROM {TABLE_NAME} LIMIT 20'),
                    conn,
                )["val"].dropna().astype(str).tolist()
            example_vals = ", ".join(distinct_vals[:10]) if distinct_vals else "—"
            lines.append(f'Column "{col}" (text). Examples: {example_vals}')
        else:
            lines.append(f'Column "{col}" ({str(dtype)})')
    return " | ".join(lines)


# ---------- MISTRAL HELPERS ----------
def _check_api_key():
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set!")


def clean_sql_output(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```sql", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```", "", s).strip()
    return s


def call_mistral_for_sql(question: str, rich_schema: str) -> str:
    _check_api_key()

    system_prompt = f"""
You are an SQL expert for a single SQLite table.

TABLE NAME: {TABLE_NAME}
SCHEMA + EXAMPLE VALUES: {rich_schema}

Rules:
- Use ONLY the table data.
- If the question cannot be answered from the table, reply: NO_ANSWER
- Handle spelling mistakes, underscores, spacing using LOWER() + REPLACE().
- Generate ONE valid SELECT query.
- No semicolons, no DDL/DML.
- Return ONLY SQL or NO_ANSWER.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    return "NO_ANSWER" if raw.upper().startswith("NO_ANSWER") else clean_sql_output(raw)


# ---------- HUMAN ANSWER ----------
def call_mistral_for_answer(question: str, sql: str, result_df: pd.DataFrame) -> str:
    _check_api_key()

    if result_df.empty:
        return "I checked the data, but there are no rows matching your question."

    preview = result_df.head(10).to_markdown(index=False)

    system_prompt = """
You are a friendly senior business analyst.

Write in a natural HUMAN tone:
- Not robotic
- Short, clear, conversational
- Example style: “There are **4 employees** working in **Chennai**.”
- Use Markdown **bold** to highlight the main facts (important numbers and key entity names).
- Do NOT mention SQL terms like COUNT, SELECT, columns, etc.
- Base answer ONLY on the result preview.
- 1–3 sentences.
"""

    user_content = f"""
Question: {question}
SQL: {sql}

Data Preview:
{preview}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    }

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------- SQL SAFETY ----------
FORBIDDEN = [r";", r"DROP", r"DELETE", r"UPDATE", r"INSERT", r"ALTER", r"CREATE"]


def is_safe_sql(sql: str) -> bool:
    if not sql.lower().startswith("select"):
        return False
    for p in FORBIDDEN:
        if re.search(p, sql, re.IGNORECASE):
            return False
    return True


def run_sql(sql: str) -> pd.DataFrame:
    if not is_safe_sql(sql):
        raise ValueError("Unsafe SQL query!")
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


# ---------- STREAMLIT APP ----------
def main():
    st.title("📊 Excel → AI SQL Chatbot")

    # persistent flags
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = existing_data_available()
    if "chat_history" not in st.session_state:
        # now we store items as {"q": "...", "a": "...", "sql": "...", "ts": "YYYY-MM-DD HH:MM:SS"}
        st.session_state["chat_history"] = []

    # upload section
    st.sidebar.header("Upload Excel / CSV")
    file = st.sidebar.file_uploader("Choose a file", type=["xlsx", "csv"])

    if file and st.sidebar.button("Load Into AI Database"):
        df = load_excel_to_db(file)
        st.session_state["data_loaded"] = True
        st.session_state["chat_history"] = []  # clear history when data replaced
        st.success("Your file has been loaded.")
        st.dataframe(df.head())

    # If there is existing data already
    if st.session_state["data_loaded"] and not file:
        st.info("Using previously uploaded data. Upload again to replace it.")

    # if no data → stop
    if not st.session_state["data_loaded"]:
        st.warning("Please upload a file to start.")
        return

    # --------------- QUESTION INPUT ----------------
    st.markdown("### 🔍 Ask a question")
    with st.form(key="ask_form"):
        question = st.text_input("Enter your question")
        submitted = st.form_submit_button("Ask AI")

    # ------------- PROCESS QUESTION ----------------
    if submitted and question.strip():
        with st.spinner("🤖 Thinking..."):
            try:
                rich_schema = get_rich_schema_description()
                sql = call_mistral_for_sql(question, rich_schema)

                if sql == "NO_ANSWER":
                    answer = "This question requires information that is not available in the uploaded data."
                    displayed_sql = None
                else:
                    displayed_sql = sql  # store for UI / history

                    # Show the generated SQL to the user inside an expander
                    with st.expander("🔎 Generated SQL (click to expand)"):
                        st.code(displayed_sql, language="sql")
                        # safety summary
                        if not is_safe_sql(displayed_sql):
                            st.error("⚠️ The generated SQL was flagged as unsafe and will NOT be executed.")
                        else:
                            st.success("SQL looks safe to execute (basic safety checks passed).")

                    # Only run if safe
                    if not is_safe_sql(displayed_sql):
                        answer = "The generated SQL was flagged as unsafe and was not executed."
                    else:
                        result_df = run_sql(displayed_sql)
                        answer = call_mistral_for_answer(question, displayed_sql, result_df)

                # append with timestamp (formatted) and SQL
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["chat_history"].append({"q": question, "a": answer, "sql": displayed_sql, "ts": ts})

                st.markdown("### 🤖 AI Answer")
                st.markdown(answer)

            except Exception as e:
                st.error(str(e))

    # --------------- CHAT HISTORY (Hidden unless clicked) ----------------
    with st.expander("📜 View Chat History (Click to Expand)"):
        history = st.session_state.get("chat_history", [])
        if history:
            # show newest first by sorting on timestamp (items without ts come last)
            def sort_key(item):
                return item.get("ts", "")
            for item in sorted(history, key=sort_key, reverse=True):
                ts_display = item.get("ts", "")
                st.markdown(f"**You ({ts_display}):** {item['q']}")
                st.markdown(f"**Bot:** {item['a']}")
                # show SQL for this item (if available) in a nested expander
                if item.get("sql"):
                    with st.expander("Show SQL for this answer"):
                        st.code(item["sql"], language="sql")
                st.markdown("---")
        else:
            st.write("No chat history yet.")


if __name__ == "__main__":
    main()
