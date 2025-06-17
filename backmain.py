import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_groq_for_spec(df: pd.DataFrame, question: str) -> str:
    """
    Ask Groq to return a JSON specification for a pie chart, based on the question.
    Returns the raw JSON string (we'll parse it downstream).
    """
    # Use a small sample or the full columns list to inform LLM
    # For simplicity, we pass column names and some sample rows.
    cols = list(df.columns)
    sample_head = df.head(5).to_dict(orient="records")  # small sample for context
    prompt = f"""
You are a data visualization assistant. The user wants a pie chart based on their question.
Return a JSON object (and only the JSON, no extra text) with exactly these fields:
- "column": the name of a column in the dataset to aggregate for the pie chart (e.g., a categorical column).
- "title": a string title for the pie chart.
- Optional fields:
  - "autopct": format string for percentages (e.g., "%1.1f%%"). If not specified, default to "%1.1f%%".
  - "figsize": list of two numbers [width, height] in inches. If not specified, default to [6, 6].
  - If the question implies filtering, you may include:
      - "filter_column": column name to filter on,
      - "filter_value": value to filter by (exact match).
  - If the question implies grouping by multiple columns for nested pie or something unusual, you can include additional instructions in "notes" field, but our code will only handle the simple single-column pie.
  
User Question:
{question}

Dataset columns: {cols}
Sample rows (first 5):
{sample_head}

Requirements:
- Return only a single JSON object, no explanation or code fences.
- Choose a categorical column appropriate for a pie chart based on the question.
- If the question explicitly names a column, use that column; otherwise choose something plausible.
- Construct a meaningful title string referencing the column or question.
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    raw = response.choices[0].message.content.strip()
    return raw

def render_pie_from_spec(df: pd.DataFrame, spec_json: str):
    """
    Parse spec_json and render a pie chart in Streamlit.
    """
    # Optionally display raw JSON for debugging in an expander
    with st.expander("Raw chart spec (JSON)"):
        st.code(spec_json, language="json")
    # Parse JSON
    try:
        spec = json.loads(spec_json)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON spec: {e}")
        return

    # Required field: column
    if "column" not in spec:
        st.error('JSON spec missing required field "column".')
        return
    column = spec["column"]
    if column not in df.columns:
        st.error(f'Column "{column}" from spec not found in DataFrame.')
        return

    # Title
    title = spec.get("title", f"Pie Chart of {column}")

    # Autopct
    autopct = spec.get("autopct", "%1.1f%%")

    # Figsize
    figsize = spec.get("figsize", [6, 6])
    # Validate figsize
    if (
        not isinstance(figsize, (list, tuple))
        or len(figsize) != 2
        or not all(isinstance(x, (int, float)) for x in figsize)
    ):
        st.warning(f'Invalid "figsize" in spec; using default [6, 6]. Got: {figsize}')
        figsize = [6, 6]

    # Optional filter
    df_to_plot = df
    if "filter_column" in spec and "filter_value" in spec:
        filt_col = spec["filter_column"]
        filt_val = spec["filter_value"]
        if filt_col not in df.columns:
            st.warning(f'filter_column "{filt_col}" not in DataFrame; ignoring filter.')
        else:
            # Simple equality filter; user may need more complex, but out of scope
            df_to_plot = df[df[filt_col] == filt_val]
            if df_to_plot.empty:
                st.warning(f'Filter on {filt_col} == {filt_val} yields no rows; using full DataFrame.')
                df_to_plot = df

    # Compute value counts
    try:
        counts = df_to_plot[column].value_counts(dropna=False)
    except Exception as e:
        st.error(f"Error computing value counts on column {column}: {e}")
        return

    if counts.empty:
        st.warning(f"No data to plot for column {column}.")
        return

    # Plot pie chart
    fig, ax = plt.subplots(figsize=figsize)
    # autopct: show percentages
    try:
        counts.plot(kind="pie", autopct=autopct, ax=ax)
    except Exception as e:
        st.error(f"Error plotting pie chart: {e}")
        return

    ax.set_ylabel("")  # usually hide the y-label for pie
    ax.set_title(title)
    # For better layout: equal aspect
    ax.axis("equal")

    st.pyplot(fig)
    plt.clf()

# App
def main():
    st.set_page_config(page_title="📊 Excel Visual Insights", layout="wide")
    st.title("🇲‌🇴‌🇹‌🇴‌🇷‌ 🇮‌🇳‌🇸‌🇺‌🇷‌🇦‌🇳‌🇨‌🇪‌ 🇩‌🇦‌🇹‌🇦‌ 🇦‌🇳‌🇦‌🇱‌🇾‌🇿‌🇪‌🇷‌")

    st.sidebar.header("Upload Excel File")
    excel_file = st.sidebar.file_uploader("Upload Excel file", type=["xls", "xlsx", "xlsb"])

    if excel_file:
        try:
            if excel_file.name.endswith(".xlsb"):
                import pyxlsb
                df = pd.read_excel(excel_file, engine="pyxlsb")
            else:
                df = pd.read_excel(excel_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df.head(20))
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

        user_question = st.text_input("Ask a question for a pie chart (e.g., Show distribution of Claim Type):")
        if user_question:
            with st.spinner("Generating pie chart spec..."):
                raw_spec = ask_groq_for_spec(df, user_question)
            # Render chart (or show errors)
            render_pie_from_spec(df, raw_spec)

if __name__ == "__main__":
    main()
