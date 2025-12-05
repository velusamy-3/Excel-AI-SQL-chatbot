import os
import re
import math
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from datetime import datetime
from io import StringIO

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
def call_mistral_for_answer(question: str, sql: str, result_df: pd.DataFrame, use_full_for_answer: bool = False) -> str:
    _check_api_key()

    if result_df.empty:
        return "I checked the data, but there are **no rows matching your question**."

    # decide what preview to send: either the full small result or a head preview
    if use_full_for_answer and len(result_df) <= 1000:
        preview_df = result_df  # send full if reasonable
    else:
        preview_df = result_df.head(10)

    preview = preview_df.to_markdown(index=False)

    system_prompt = """
You are a friendly senior business analyst.

Write in a natural HUMAN tone:
- Not robotic
- Short, clear, conversational
- Example style: “There are **4 employees** working in **Chennai**.”
- Use Markdown **bold** to highlight the main facts (important numbers and key entity names).
- Do NOT mention SQL terms like COUNT, SELECT, columns, etc.
- Base answer ONLY on the result preview.
- 1–4 sentences. If the user asked to show the full dataset, explicitly state that you displayed the full data or provided a download link.
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


# ---------- Helpers for "show all" behavior ----------
def is_display_all_intent(question: str, sql: str) -> bool:
    """
    Detects if user intent is to display the full dataset.
    Checks both natural language cues and whether SQL is a bare SELECT * FROM table.
    """
    q = question.lower()
    display_triggers = [
        "show all", "display all", "display the whole", "show the whole",
        "full dataset", "all the data", "every row", "give me the data",
        "show entire", "show me everything", "list all", "dump all"
    ]
    if any(trigger in q for trigger in display_triggers):
        return True

    # simple check: SELECT * FROM TABLE_NAME (optionally with double quotes)
    normalized_sql = sql.lower().replace('"', "").replace("'", "").strip()
    if re.match(rf"^select\s+\*\s+from\s+{re.escape(TABLE_NAME.lower())}\s*$", normalized_sql):
        return True

    return False


def dataframe_summary(df: pd.DataFrame, top_n_vals: int = 5) -> str:
    rows, cols = df.shape
    summary_lines = [f"**Rows:** **{rows}**  •  **Columns:** **{cols}**"]
    # numeric stats (brief)
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        desc = num_df.describe().T
        # take first 3 numeric columns to summarize (avoid long output)
        for i, (col, row) in enumerate(desc.iterrows()):
            if i >= 3:
                break
            summary_lines.append(f"**{col}** — mean: {round(row['mean'], 2)}, std: {round(row['std'],2)}")
    # top values for up to 3 text cols
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()[:3]
    for col in text_cols:
        top_vals = df[col].dropna().astype(str).value_counts().head(top_n_vals)
        if not top_vals.empty:
            vals = "; ".join([f"{v} ({c})" for v, c in zip(top_vals.index.tolist(), top_vals.values.tolist())])
            summary_lines.append(f"**{col}** top: {vals}")
    return "  \n".join(summary_lines)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="Excel → AI SQL Chatbot", layout="wide")
    st.title("📊 Excel → AI SQL Chatbot")

    # persistent flags
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = existing_data_available()
    if "chat_history" not in st.session_state:
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
                    result_df = pd.DataFrame()
                    show_full_intent = False
                else:
                    displayed_sql = sql  # store for UI / history

                    # Show the generated SQL to the user inside an expander
                    with st.expander("🔎 Generated SQL (click to expand)"):
                        st.code(displayed_sql, language="sql")
                        # safety summary
                        if not is_safe_sql(displayed_sql):
                            st.error("⚠️ The generated SQL was flagged as unsafe and will NOT be executed.")
                            result_df = pd.DataFrame()
                            show_full_intent = False
                        else:
                            st.success("SQL looks safe to execute (basic safety checks passed).")
                            # determine intent to show all
                            show_full_intent = is_display_all_intent(question, displayed_sql)
                            result_df = run_sql(displayed_sql)

                    # Only run if safe (result_df set above)
                    if result_df is None:
                        result_df = pd.DataFrame()

                    # If user intended to see all data or to preview, produce a professional display
                    # Safety: if very large, warn and limit initial render
                    MAX_RENDER_FULL_ROWS = 20000  # threshold for full immediate render
                    PREVIEW_ROWS = 200

                    if show_full_intent:
                        # create summary
                        summary_text = dataframe_summary(result_df)
                        st.markdown("### 📋 Dataset Summary")
                        st.markdown(summary_text)

                        # offer download always
                        csv_bytes = df_to_csv_bytes(result_df)
                        st.download_button(
                            "⬇️ Download full result as CSV",
                            data=csv_bytes,
                            file_name="query_result.csv",
                            mime="text/csv",
                        )

                        # If result is huge, warn and show preview + explicit button to render full
                        total_rows = len(result_df)
                        if total_rows > MAX_RENDER_FULL_ROWS:
                            st.warning(
                                f"The result contains **{total_rows:,}** rows. "
                                "Rendering all rows in the browser may be slow. "
                                "You can download the full CSV above, or render a preview below."
                            )
                            st.dataframe(result_df.head(PREVIEW_ROWS))
                            if st.button("Show full table anyway (may be slow)"):
                                st.dataframe(result_df)
                        else:
                            # safe to display full result immediately if small/moderate
                            # but still put inside an expander so UI stays tidy
                            with st.expander(f"Show full table ({total_rows} rows)"):
                                st.dataframe(result_df)

                        # generate the AI answer using full data if it's not too large
                        use_full_for_answer = total_rows <= 1000  # cap for LLM preview
                        answer = call_mistral_for_answer(question, displayed_sql, result_df, use_full_for_answer)

                    else:
                        # normal behavior: show a concise preview and let user expand
                        if result_df.empty:
                            answer = "I checked the data, but there are no rows matching your question."
                        else:
                            st.markdown("### 🔢 Query Result Preview")
                            st.dataframe(result_df.head(10))
                            # offer download of the result
                            csv_bytes = df_to_csv_bytes(result_df)
                            st.download_button(
                                "⬇️ Download query result as CSV",
                                data=csv_bytes,
                                file_name="query_result.csv",
                                mime="text/csv",
                            )

                            answer = call_mistral_for_answer(question, displayed_sql, result_df, use_full_for_answer=False)

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
