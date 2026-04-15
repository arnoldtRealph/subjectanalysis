import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from docx import Document
from docx.shared import Inches
from io import BytesIO

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Saul Damon High School | Term Analysis",
    layout="wide",
    page_icon="📊"
)

# ====================== LIGHT STYLING ======================
st.markdown("""
<style>
.stApp {background-color:#F5F7FA; color:#1F2937; font-family:Arial;}
.main-title {font-size:42px; text-align:center; font-weight:bold;}
.sub-title {font-size:22px; text-align:center; margin-bottom:30px; color:#4B5563;}
.section-header {font-size:22px; font-weight:600; color:#2563EB; margin-top:30px;}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"])
    chart_type = st.selectbox("Chart Type", ["Bar", "Stacked Bar", "Pie"])

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("Upload your Excel file to begin.")
    st.stop()

# ====================== DATA ======================
@st.cache_data
def process_data(file):
    df_raw = pd.read_excel(file, header=None)

    grade_starts = df_raw.index[df_raw[0].str.contains("GRADE", na=False)].tolist()
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_dfs = {}

    for i, grade in enumerate(grades):
        if i >= len(grade_starts):
            break

        start_idx = grade_starts[i] + 2
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df_raw)

        gdf = df_raw.iloc[start_idx:end_idx].dropna(how="all").reset_index(drop=True)

        cols = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3",
                "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        gdf = gdf.iloc[:, :len(cols)]
        gdf.columns = cols

        gdf["SUBJECT"] = gdf["SUBJECT"].astype(str).str.strip()
        gdf = gdf[gdf["SUBJECT"] != ""]

        numeric_cols = ["AVERAGE MARK"] + [f"LEVEL {i}" for i in range(1, 8)] + ["TOTAL"]
        gdf[numeric_cols] = gdf[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        gdf["TOTAL"] = gdf[level_cols].sum(axis=1)

        grade_dfs[grade] = gdf

    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== PIE FIX ======================
def autopct_format(pct):
    return f"{pct:.1f}%" if pct > 6 else ""

# ====================== CHART IMAGE ======================
def create_pie_chart(gdf, title):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(
        gdf["AVERAGE MARK"],
        labels=gdf["SUBJECT"],
        autopct=autopct_format,
        startangle=90,
        colors=sns.color_palette("pastel")
    )
    ax.set_title(title)
    ax.axis('equal')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)
    return img

# ====================== REPORT ======================
def generate_report(grade_dfs, term):
    doc = Document()
    doc.add_heading(f"Saul Damon High School\n{term} Report", 0)

    for grade, gdf in grade_dfs.items():
        if gdf.empty:
            continue

        doc.add_heading(grade, 1)

        doc.add_paragraph(f"Average Mark: {gdf['AVERAGE MARK'].mean():.1f}%")
        doc.add_paragraph(f"Total Learners: {int(gdf['TOTAL'].sum())}")

        chart_img = create_pie_chart(gdf, f"{grade} Average Marks")
        doc.add_picture(chart_img, width=Inches(5.5))

        table = doc.add_table(rows=1, cols=3)
        headers = table.rows[0].cells
        headers[0].text = "Subject"
        headers[1].text = "Average"
        headers[2].text = "Learners"

        for _, row in gdf.iterrows():
            cells = table.add_row().cells
            cells[0].text = str(row["SUBJECT"])
            cells[1].text = f"{row['AVERAGE MARK']:.1f}%"
            cells[2].text = str(int(row["TOTAL"]))

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# ====================== DOWNLOAD FULL ======================
full_report = generate_report(grade_dfs, term)

st.download_button(
    "📄 Download Full School Report (With Charts)",
    data=full_report,
    file_name=f"{term}_Full_Report.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

# ====================== DASHBOARD ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty:
        continue

    st.markdown(f'<p class="section-header">{grade}</p>', unsafe_allow_html=True)

    # Grade download
    grade_report = generate_report({grade: gdf}, term)
    st.download_button(
        f"Download {grade} Report (With Charts)",
        data=grade_report,
        file_name=f"{grade}_{term}.docx"
    )

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", len(gdf))
    c2.metric("Avg Mark", f"{gdf['AVERAGE MARK'].mean():.1f}%")
    c3.metric("Learners", int(gdf["TOTAL"].sum()))

    # Table
    st.dataframe(gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]], use_container_width=True)

    # ====================== MAIN CHART ======================
    st.subheader("Average Marks")

    fig, ax = plt.subplots(figsize=(7,7))

    if chart_type == "Pie":
        ax.pie(
            gdf["AVERAGE MARK"],
            labels=gdf["SUBJECT"],
            autopct=autopct_format,
            startangle=90,
            colors=sns.color_palette("pastel")
        )
        ax.axis('equal')

    elif chart_type == "Bar":
        sns.barplot(x="AVERAGE MARK", y="SUBJECT", data=gdf, color="#3B82F6", ax=ax)

    else:
        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        gdf.set_index("SUBJECT")[level_cols].plot(kind="barh", stacked=True, ax=ax, colormap="Blues")

    st.pyplot(fig)

    # ====================== LEVEL DISTRIBUTION ======================
    st.markdown('<p class="section-header">Level Distribution</p>', unsafe_allow_html=True)

    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    cols = st.columns(3)

    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 3]:
            levels = row[level_cols]

            if levels.sum() == 0:
                st.write("No data")
                continue

            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie(
                levels,
                autopct=autopct_format,
                startangle=90,
                colors=sns.color_palette("Blues")
            )
            ax.set_title(row["SUBJECT"], fontsize=10)
            ax.axis('equal')
            st.pyplot(fig)

    # ====================== PASS / FAIL ======================
    st.markdown('<p class="section-header">Pass / Fail</p>', unsafe_allow_html=True)

    gdf["FAILED"] = gdf["LEVEL 1"]
    gdf["PASSED"] = gdf["TOTAL"] - gdf["FAILED"]

    fig, ax = plt.subplots(figsize=(10,5))
    gdf.set_index("SUBJECT")[["FAILED","PASSED"]].plot(kind="barh", stacked=True, ax=ax)
    st.pyplot(fig)

    # ====================== INSIGHTS ======================
    st.markdown('<p class="section-header">Insights</p>', unsafe_allow_html=True)

    for _, row in gdf.iterrows():
        if row["TOTAL"] == 0:
            continue

        fail_rate = (row["FAILED"] / row["TOTAL"]) * 100

        if fail_rate > 30:
            st.error(f"{row['SUBJECT']}: High fail rate ({fail_rate:.1f}%)")
        elif row["AVERAGE MARK"] < 50:
            st.warning(f"{row['SUBJECT']}: Low average ({row['AVERAGE MARK']:.1f}%)")
        else:
            st.success(f"{row['SUBJECT']}: Performing well")

st.caption("Saul Damon High School • Professional Dashboard")
