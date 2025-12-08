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

    # 🔹 Try to convert numeric-looking columns to real numbers
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

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
            with engine.connect() as conn2:
                distinct_vals = pd.read_sql(
                    text(f'SELECT DISTINCT "{col}" AS val FROM {TABLE_NAME} LIMIT 20'),
                    conn2,
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


def _call_mistral(payload: dict) -> str:
    """Shared helper to call Mistral with basic error handling."""
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

    resp = requests.post(MISTRAL_API_URL, json=payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(f"Mistral API error: {resp.status_code} - {resp.text}")

    data = resp.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected Mistral response: {data}")

    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Mistral content: {content}")

    return content.strip()


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

Your job: turn the user's natural-language question into exactly ONE valid
SELECT query on {TABLE_NAME}.

Very important rules:

1) Table & columns
- Use ONLY the table {TABLE_NAME}.
- Use ONLY the columns that appear in the schema description above.
- When the user mentions a field like "stock quantity", "qty in stock",
  "available qty", etc., map it to the CLOSEST column name in the schema.
- Treat spaces, underscores, hyphens, and case as irrelevant when matching
  names. (Example: "stock_quantity", "Stock Quantity", "stock-quantity"
  should all be treated as the same concept.)

2) Comparison / filter questions  (THIS IS CRITICAL)
- If the user asks things like:
    - "greater than 500", "more than 500", "over 500", "above 500"
      → use the operator ">".
    - "greater than or equal to 500", "at least 500", "no less than 500"
      → use ">=".
    - "less than 500", "under 500", "below 500"
      → use "<".
    - "less than or equal to 500", "at most 500", "no more than 500"
      → use "<=".
    - "between 100 and 200"
      → use "BETWEEN 100 AND 200" (inclusive).

- ALWAYS extract the numeric threshold(s) from the question and use them
  literally in the WHERE clause.

- If the target column is text in the schema (because the CSV had strings),
  still perform the numeric comparison by casting to REAL. For example:

    CAST("stock quantity" AS REAL) > 500

  or

    CAST("price" AS REAL) BETWEEN 100 AND 200

3) General SQL constraints
- Generate exactly ONE valid SELECT query on {TABLE_NAME}.
- No semicolons.
- No PRAGMA, no INFORMATION_SCHEMA, no system/metadata tables.
- Do NOT try to inspect the schema using SQL; rely ONLY on the description.
- No DDL or DML: do NOT use CREATE, DROP, INSERT, UPDATE, DELETE, ALTER, etc.
- Do not alias the table or join with anything else.

4) Output format
- Return ONLY the raw SQL text, with no explanation.
- If and only if the question truly cannot be answered from this single
  table (for example, it clearly needs another system or external data),
  then return exactly: NO_ANSWER

Example-style mapping (these are just examples, do not echo them back):

User: "List all products where stock quantity is greater than 500"
SQL:  SELECT * FROM {TABLE_NAME}
      WHERE CAST("stock quantity" AS REAL) > 500

User: "Show customers with total amount less than 1000"
SQL:  SELECT * FROM {TABLE_NAME}
      WHERE CAST("total amount" AS REAL) < 1000
"""

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    }

    raw = _call_mistral(payload)

    return "NO_ANSWER" if raw.upper().startswith("NO_ANSWER") else clean_sql_output(raw)


# ---------- HUMAN ANSWER ----------
def call_mistral_for_answer(
    question: str,
    sql: str,
    result_df: pd.DataFrame,
    use_full_for_answer: bool = False
) -> str:
    _check_api_key()

    if result_df.empty:
        return "I checked the data, but there are **no rows matching your question**."

    if use_full_for_answer and len(result_df) <= 1000:
        preview_df = result_df
    else:
        preview_df = result_df.head(10)

    preview = preview_df.to_markdown(index=False)

    row_count = len(result_df)
    col_count = result_df.shape[1]

    system_prompt = """
You are a friendly senior business analyst.

Write in a natural HUMAN tone:
- Not robotic
- Short, clear, conversational
- Example style: “There are **4 employees** working in **Chennai**.”
- Use Markdown **bold** to highlight the main facts (important numbers and key entity names).
- Do NOT mention SQL terms like COUNT, SELECT, columns, etc.
- Base answer ONLY on the result preview and the row/column counts provided.
- If the user asks “how many / number of / count of…”, use the **Rows** metadata instead of guessing from the preview.
- 1–4 sentences. If the user asked to show the full dataset, explicitly state that you displayed the full data or provided a download link.
"""

    user_content = f"""
Question: {question}
SQL: {sql}

Result metadata:
- Rows: {row_count}
- Columns: {col_count}

Data Preview:
{preview}
"""

    payload = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    }

    return _call_mistral(payload)


# ---------- SQL SAFETY ----------

def is_safe_sql(sql: str) -> bool:
    """Stricter SQL safety for real-company use."""
    if not isinstance(sql, str):
        return False

    s = sql.strip()
    s_lower = s.lower()

    # Must be a SELECT
    if not s_lower.startswith("select"):
        return False

    # Must query ONLY our table (no other FROM targets)
    from_pattern = rf"\bfrom\s+{re.escape(TABLE_NAME.lower())}\b"
    if re.search(from_pattern, s_lower) is None:
        return False

    # Forbid dangerous patterns / keywords
    forbidden_patterns = [
        r";",                  # no multiple statements
        r"\bunion\b",
        r"\bjoin\b",
        r"\bwith\b",
        r"\battach\b",
        r"\bsqlite_master\b",
        r"\binformation_schema\b",
        r"\bpragma\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\binsert\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\bcreate\b",
    ]

    for pat in forbidden_patterns:
        if re.search(pat, s_lower, re.IGNORECASE):
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
    q = question.lower()
    display_triggers = [
        "show all", "display all", "display the whole", "show the whole",
        "full dataset", "all the data", "every row", "give me the data",
        "show entire", "show me everything", "list all", "dump all"
    ]
    if any(trigger in q for trigger in display_triggers):
        return True

    normalized_sql = sql.lower().replace('"', "").replace("'", "").strip()
    if re.match(rf"^select\s+\*\s+from\s+{re.escape(TABLE_NAME.lower())}\s*$", normalized_sql):
        return True

    return False


def dataframe_summary(df: pd.DataFrame, top_n_vals: int = 5) -> str:
    rows, cols = df.shape
    summary_lines = [f"**Rows:** **{rows}**  •  **Columns:** **{cols}**"]
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        desc = num_df.describe().T
        for i, (col, row) in enumerate(desc.iterrows()):
            if i >= 3:
                break
            summary_lines.append(f"**{col}** — mean: {round(row['mean'], 2)}, std: {round(row['std'], 2)}")
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()[:3]
    for col in text_cols:
        top_vals = df[col].dropna().astype(str).value_counts().head(top_n_vals)
        if not top_vals.empty:
            vals = "; ".join([f"{v} ({c})" for v, c in zip(top_vals.index.tolist(), top_vals.values.tolist())])
            summary_lines.append(f"**{col}** top: {vals}")
    return "  \n".join(summary_lines)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------- EXACT LOOKUP HELPERS ----------
def normalize_text(s: str) -> str:
    """Lowercase and remove non-alphanumeric to compare column names / IDs."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def handle_exact_lookup(question: str):
    """
    Handles ID-based questions like:
      - "What is the order status of PO-00076?"
      - "Give details of PO-00034 and PO-00037"

    Supports MULTIPLE IDs in one question.

    Returns:
      (result_df, direct_answer, used_sql, id_column)
      - If can't detect ID pattern -> None
      - If IDs not found in data -> empty df + direct_answer message
      - If rows found          -> result_df, direct_answer=None
    """
    q = question.strip()
    if not q:
        return None

    # No spaces allowed → avoids matching phrases like "than 500"
    id_pattern = re.compile(r"\b[A-Za-z]{2,15}[-_]?\d{1,10}\b")
    id_matches = id_pattern.findall(q)
    if not id_matches:
        return None

    # unique IDs, preserve order
    unique_ids = list(dict.fromkeys(id_matches))
    norm_targets = {normalize_text(i) for i in unique_ids}

    engine = get_engine()
    with engine.connect() as conn:
        full_df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)

    text_cols = full_df.select_dtypes(include=["object"]).columns.tolist()
    if not text_cols:
        return None

    target_df = None
    used_col = None

    # find the first text column where any normalized value matches IDs
    for col in text_cols:
        col_norm = full_df[col].astype(str).map(normalize_text)
        mask = col_norm.isin(norm_targets)
        if mask.any():
            target_df = full_df[mask].copy()
            used_col = col
            break

    if target_df is None or target_df.empty:
        ids_list = ", ".join(f"**{i}**" for i in unique_ids)
        answer = f"I couldn't find any rows with IDs {ids_list} in the data."
        used_sql = "-- exact lookup: no rows matched normalized IDs"
        return pd.DataFrame(), answer, used_sql, used_col

    used_sql = f"-- exact lookup performed in Python on column '{used_col}' for normalized IDs {list(norm_targets)}"
    # let caller build final answer (single-row or multi-row)
    return target_df, None, used_sql, used_col


# ---------- SINGLE / MULTI ROW ANSWER HELPERS ----------
def build_answer_for_single_row(question: str, result_df: pd.DataFrame) -> str:
    """
    Build a deterministic human-style answer when there is exactly ONE row.
    - If question says "details", returns bullet list of all fields.
    - Otherwise, returns one structured sentence with requested fields.
    """
    row = result_df.iloc[0]
    q_norm = question.lower()

    # detect ID
    id_pattern = re.compile(r"\b[A-Za-z]{2,15}[-_]?\d{1,10}\b")
    id_match = id_pattern.search(question)
    id_val = id_match.group(0) if id_match else None

    # pick columns mentioned in question
    selected_cols = []
    for col in result_df.columns:
        col_norm = str(col).lower().replace("_", " ").strip()
        if col_norm and col_norm in q_norm:
            selected_cols.append(col)

    # if none matched, treat as "details" of full row
    if not selected_cols:
        selected_cols = list(result_df.columns)

    # if user explicitly says "detail", show bullet list
    if "detail" in q_norm:
        header = f"Here are the details for **{id_val}**:" if id_val else "Here are the details for this record:"
        lines = []
        for col in selected_cols:
            nice_col = str(col).replace("_", " ").title()
            val = row[col]
            lines.append(f"- **{nice_col}**: **{val}**")
        return header + "\n" + "\n".join(lines)

    # normal sentence
    fragments = []
    for col in selected_cols:
        nice_col = str(col).replace("_", " ").title()
        val = row[col]
        fragments.append(f"**{nice_col}** is **{val}**")

    if len(fragments) == 1:
        core = fragments[0]
    else:
        *first_parts, last_part = fragments
        core = ", ".join(first_parts) + f", and {last_part}"

    if id_val:
        return f"For **{id_val}**, {core}."
    else:
        return core + "."


def build_answer_for_multi_rows(
    question: str,
    result_df: pd.DataFrame,
    id_col: str | None = None
) -> str:
    """
    Build a structured answer when MULTIPLE rows are returned (e.g. multiple POs).
    Produces bullet list per ID.
    """
    q_norm = question.lower()

    # columns mentioned in question
    selected_cols = []
    for col in result_df.columns:
        col_norm = str(col).lower().replace("_", " ").strip()
        if col_norm and col_norm in q_norm:
            selected_cols.append(col)

    if not selected_cols:
        # if ask "details" -> all columns; else still all to be simple
        selected_cols = list(result_df.columns)

    header = "Here are the details you asked for:\n"
    lines = []

    if id_col and id_col in result_df.columns:
        for _, r in result_df.iterrows():
            id_val = r[id_col]
            parts = []
            for col in selected_cols:
                nice_col = str(col).replace("_", " ").title()
                val = r[col]
                parts.append(f"**{nice_col}**: **{val}**")
            joined = ", ".join(parts)
            lines.append(f"- For **{id_val}**: {joined}.")
    else:
        # fallback if we don't know the ID column
        for idx, r in result_df.iterrows():
            parts = []
            for col in selected_cols:
                nice_col = str(col).replace("_", " ").title()
                val = r[col]
                parts.append(f"**{nice_col}**: **{val}**")
            joined = ", ".join(parts)
            lines.append(f"- Row {idx + 1}: {joined}.")

    return header + "\n".join(lines)


# ---------- STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="📊 Excel → AI SQL Chatbot", layout="wide")
    st.title("📊 Excel → AI SQL Chatbot")
    st.caption("Upload an Excel/CSV file and ask questions in plain English. The bot converts your question → SQL → human answer.")

    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = existing_data_available()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # -------- Sidebar: Upload & dataset info --------
    with st.sidebar:
        st.header("📁 Data Upload")
        file = st.file_uploader("Choose a file", type=["xlsx", "csv"])

        if file and st.button("📥 Load Into AI Database"):
            df = load_excel_to_db(file)
            st.session_state["data_loaded"] = True
            st.session_state["chat_history"] = []
            st.success("✅ Your file has been loaded.")
            st.dataframe(df.head())
        
        if st.session_state["data_loaded"] and not file:
            st.info("Using previously uploaded data. Upload again to replace it.")

        st.markdown("---")
        st.subheader("🧬 Table Sample")
        if st.session_state["data_loaded"]:
            try:
                with get_engine().connect() as conn:
                    sample = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 5", conn)
                st.dataframe(sample)
            except Exception as e:
                st.error(f"Error reading sample: {e}")
        else:
            st.caption("Upload a file to see a preview here.")

    # -------- Main layout: query, answer, history --------
    st.markdown("### 🔍 Ask a question")
    st.write(
        "Examples: `Show all data`, `how many records are there`, "
        "`List all products where stock quantity is greater than 500`"
    )

    with st.form(key="ask_form"):
        question = st.text_input("Type your question here:")
        submitted = st.form_submit_button("Ask AI 💬")

    if not st.session_state["data_loaded"]:
        st.warning("⚠️ Please upload a file to start.")
        return

    if submitted and question.strip():
        with st.spinner("🤖 Thinking..."):
            try:
                # ---------- 1) Exact lookup path (IDs like PO-00034) ----------
                direct_result = handle_exact_lookup(question)
                if direct_result is not None:
                    result_df, direct_answer, displayed_sql, id_col = direct_result

                    if result_df is not None and not result_df.empty:
                        st.markdown("### 🔢 Query Result")
                        st.dataframe(result_df.head(10))
                        csv_bytes = df_to_csv_bytes(result_df)
                        st.download_button(
                            "⬇️ Download result as CSV",
                            data=csv_bytes,
                            file_name="query_result.csv",
                            mime="text/csv",
                        )

                        if direct_answer:
                            # only used for "no rows" case normally
                            answer = direct_answer
                        else:
                            if len(result_df) == 1:
                                answer = build_answer_for_single_row(question, result_df)
                            else:
                                answer = build_answer_for_multi_rows(
                                    question,
                                    result_df,
                                    id_col=id_col,
                                )
                    else:
                        answer = direct_answer or "I couldn't find any matching rows."

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["chat_history"].append(
                        {"q": question, "a": answer, "sql": displayed_sql, "ts": ts}
                    )

                    st.markdown("### 🤖 AI Answer")
                    st.markdown(answer)
                    return  # skip Mistral SQL flow

                # ---------- 2) Normal Mistral -> SQL flow (NO conversation memory) ----------
                rich_schema = get_rich_schema_description()
                sql = call_mistral_for_sql(question, rich_schema)

                if sql == "NO_ANSWER":
                    answer = "This question requires information that is not available in the uploaded data."
                    displayed_sql = None
                    result_df = pd.DataFrame()
                    show_full_intent = False
                else:
                    displayed_sql = sql
                    # (SQL is not shown in UI)
                    if not is_safe_sql(displayed_sql):
                        st.error("⚠️ The generated SQL was flagged as unsafe and will NOT be executed.")
                        result_df = pd.DataFrame()
                        show_full_intent = False
                    else:
                        show_full_intent = is_display_all_intent(question, displayed_sql)
                        result_df = run_sql(displayed_sql)

                if result_df is None:
                    result_df = pd.DataFrame()

                MAX_RENDER_FULL_ROWS = 20000
                PREVIEW_ROWS = 200

                if show_full_intent:
                    summary_text = dataframe_summary(result_df)
                    st.markdown("### 📋 Dataset Summary")
                    st.markdown(summary_text)

                    csv_bytes = df_to_csv_bytes(result_df)
                    st.download_button(
                        "⬇️ Download full result as CSV",
                        data=csv_bytes,
                        file_name="query_result.csv",
                        mime="text/csv",
                    )

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
                        with st.expander(f"Show full table ({total_rows} rows)"):
                            st.dataframe(result_df)

                    if total_rows == 1:
                        answer = build_answer_for_single_row(question, result_df)
                    else:
                        use_full_for_answer = total_rows <= 1000
                        answer = call_mistral_for_answer(
                            question,
                            displayed_sql,
                            result_df,
                            use_full_for_answer,
                        )

                else:
                    if result_df.empty:
                        answer = "I checked the data, but there are no rows matching your question."
                    else:
                        st.markdown("### 🔢 Query Result Preview")
                        st.dataframe(result_df.head(10))

                        csv_bytes = df_to_csv_bytes(result_df)
                        st.download_button(
                            "⬇️ Download query result as CSV",
                            data=csv_bytes,
                            file_name="query_result.csv",
                            mime="text/csv",
                        )

                        if len(result_df) == 1:
                            answer = build_answer_for_single_row(question, result_df)
                        else:
                            answer = call_mistral_for_answer(
                                question,
                                displayed_sql,
                                result_df,
                                use_full_for_answer=False,
                            )

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["chat_history"].append(
                    {"q": question, "a": answer, "sql": displayed_sql, "ts": ts}
                )

                st.markdown("### 🤖 AI Answer")
                st.markdown(answer)

            except Exception as e:
                st.error(str(e))

    # -------- Chat History --------
    with st.expander("📜 View Chat History"):
        history = st.session_state.get("chat_history", [])
        if history:
            def sort_key(item):
                return item.get("ts", "")
            for item in sorted(history, key=sort_key, reverse=True):
                st.markdown(f"**You:** {item['q']}")
                st.markdown(f"**Bot:** {item['a']}")
                # SQL is NOT displayed in history UI
                st.markdown("---")
        else:
            st.write("No chat history yet.")


if __name__ == "__main__":
    main()
