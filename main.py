import streamlit as st
import pandas as pd
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
import plotly.express as px

def inject_css_for_text_input():
    st.markdown(
        """
    <style>
    input[type="text"] {
        background-color: #fffbea !important;
        border: 2px solid #ffcc00 !important;
        border-radius: 5px !important;
        padding: 8px !important;
        color: #000000 !important;
    }
    input[type="text"]::placeholder {
        color: #666666 !important;
    }
    input[type="text"]:focus {
        background-color: #ffffff !important;
        border: 2px solid #ff9900 !important;
        box-shadow: 0 0 5px #ff9900 !important;
        outline: none !important;
    }
    label {
        font-weight: bold;
        color: #333333;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_json_from_response(response: str) -> str:
    response = re.sub(r"```(?:json)?", "", response, flags=re.IGNORECASE).strip("` \n")
    return response

def safe_json_load(json_str: str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.warning(f"Attempting to fix minor JSON issues: {e}")
        s = json_str.replace('\n', '').replace('\r', '').replace('\t', '')
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        try:
            return json.loads(s)
        except Exception as e2:
            st.error(f"❌ Still failed to parse JSON after fixes: {e2}")
            return None

def generate_prompt(df: pd.DataFrame, question: str) -> str:
    cols = list(df.columns)
    sample_data = df.sample(min(20, len(df))).to_dict(orient="records")
    col_values = {
        col: df[col].dropna().unique().tolist()[:10]
        for col in df.select_dtypes(include="object")
    }
    prompt = f"""
You are a professional data analysis and chart assistant.

You will return a JSON object with two keys:
1. "description": A clear and concise explanation of the insights based on the user’s question and the dataset. Use natural language and analytical reasoning. Describe why the chart type is chosen, what trend or pattern it helps visualize, and summarize key observations if possible.
2. 'chart': A valid chart spec used to generate a chart using Plotly.

Supported chart types: ["pie", "bar", "line", "scatter", "histogram"]

Chart format:
{{
  "chart_type": "...",
  "column": "...",
  "value_column": "...",
  "title": "...",
  "autopct": "...",
  "figsize": [6, 6],
  "filter_column": "...",
  "filter_value": "..."
}}

Return valid JSON only. Do NOT include markdown/code fences. No extra text outside the JSON.

Dataset columns: {cols}
Sample data: {sample_data}
User question: {question}
"""
    return prompt

def ask_groq_for_spec(df: pd.DataFrame, question: str) -> dict:
    prompt = generate_prompt(df, question)
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
    except Exception as e:
        st.error(f"❌ Error calling Groq API: {e}")
        return {"error": "Failed to get chart spec from LLM."}

    raw_response = response.choices[0].message.content.strip()
    clean_json = extract_json_from_response(raw_response)
    parsed = safe_json_load(clean_json)
    if parsed is None:
        return {"error": "Failed to parse combined JSON (description + chart)."}
    return parsed

def render_chart_plotly(df: pd.DataFrame, spec: dict):
    chart = spec.get("chart", {})
    chart_type = chart.get("chart_type")
    column = chart.get("column")
    value_column = chart.get("value_column")
    title = chart.get("title", "Generated Chart")
    figsize = chart.get("figsize", [6, 6])
    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    df_plot = df
    if chart.get("filter_column") and chart.get("filter_value") is not None:
        fc, fv = chart["filter_column"], chart["filter_value"]
        if fc in df.columns:
            df_filtered = df[df[fc] == fv]
            if not df_filtered.empty:
                df_plot = df_filtered

    try:
        fig = None
        if chart_type == "pie":
            if value_column and value_column in df_plot.columns:
                agg = df_plot.groupby(column)[value_column].sum().reset_index()
                fig = px.pie(agg, names=column, values=value_column, title=title)
            else:
                counts = df_plot[column].value_counts().reset_index()
                counts.columns = [column, "count"]
                fig = px.pie(counts, names=column, values="count", title=title)
        elif chart_type == "bar":
            if value_column and value_column in df_plot.columns:
                agg = df_plot.groupby(column)[value_column].mean().reset_index()
                fig = px.bar(agg, x=column, y=value_column, title=title)
            else:
                counts = df_plot[column].value_counts().reset_index()
                counts.columns = [column, "count"]
                fig = px.bar(counts, x=column, y="count", title=title)
        elif chart_type == "line":
            if value_column and value_column in df_plot.columns:
                agg = df_plot.groupby(column)[value_column].mean().reset_index()
                fig = px.line(agg, x=column, y=value_column, title=title)
            else:
                counts = df_plot[column].value_counts().sort_index().reset_index()
                counts.columns = [column, "count"]
                fig = px.line(counts, x=column, y="count", title=title)
        elif chart_type == "scatter":
            df_sc = df_plot[[column, value_column]].dropna()
            fig = px.scatter(df_sc, x=column, y=value_column, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(df_plot, x=column, title=title)
        else:
            st.error("Unsupported chart type.")
            return

        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Chart rendering failed: {e}")

def main():
    st.set_page_config(page_title="🇲‌🇴‌🇹‌🇴‌🇷‌ 🇮‌🇳‌🇸‌🇺‌🇷‌🇦‌🇳‌🇨‌🇪‌ 🇩‌🇦‌🇹‌🇦‌ 🇦‌🇳‌🇦‌🇱‌🇾‌🇿‌🇪‌🇷‌", layout="wide")
    # Hide Streamlit file upload size limit message
    st.markdown("""
        <style>
        [data-testid="stFileUploader"] small {
            display: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("🇲‌🇴‌🇹‌🇴‌🇷‌ 🇮‌🇳‌🇸‌🇺‌🇷‌🇦‌🇳‌🇨‌🇪‌ 🇩‌🇦‌🇹‌🇦‌ 🇦‌🇳‌🇦‌🇱‌🇾‌🇿‌🇪‌🇷‌")

    inject_css_for_text_input()

    st.sidebar.header("Upload Excel File")
    excel_file = st.sidebar.file_uploader("Upload your Excel file", type=["xls", "xlsx", "xlsb"])

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
            st.error(f"❌ Failed to read file: {e}")
            return

        user_question = st.text_input("Ask your question about the data:", placeholder="e.g., Show me claim amount by region")
        if user_question:
            with st.spinner("🧠 Analyzing and generating chart..."):
                spec = ask_groq_for_spec(df, user_question)

            if "error" in spec:
                st.error(spec["error"])
            else:
                description = spec.get("description", "No description provided.")
                st.subheader("𝘋𝘢𝘵𝘢 𝘐𝘯𝘴𝘪𝘨𝘩𝘵")
                st.write(description)
                st.subheader("𝘝𝘪𝘴𝘶𝘢𝘭𝘪𝘻𝘢𝘵𝘪𝘰𝘯")
                render_chart_plotly(df, spec)
    # else:
    #     st.info("📂 Please upload an Excel file to begin.")

if __name__ == "__main__":
    main()
