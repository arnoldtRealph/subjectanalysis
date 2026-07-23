import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from io import BytesIO

# ====================== CONFIG ======================
st.set_page_config(page_title="Saul Damon High School | Term Analysis", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stApp {background-color:#F5F7FA; color:#1F2937;}
    .main-title {font-size:42px; text-align:center; font-weight:bold; color:#1E3A8A;}
    .sub-title {font-size:24px; text-align:center; margin-bottom:30px; color:#4B5563;}
    .section-header {font-size:24px; font-weight:700; color:#1E40AF; margin:25px 0 10px 0;}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE (Excel)", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"])
    chart_type = st.selectbox("Main Chart Type", ["Bar", "Stacked Bar", "Pie"])

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("Upload your Excel file to begin.")
    st.stop()

# ====================== DATA PROCESSING ======================
@st.cache_data
def process_data(file):
    df_raw = pd.read_excel(file, header=None)
    grade_starts = df_raw.index[df_raw[0].astype(str).str.contains("GRADE", na=False, case=False)].tolist()
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_dfs = {}

    for i, grade in enumerate(grades):
        if i >= len(grade_starts):
            break
        start_idx = grade_starts[i] + 2
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df_raw)
        
        gdf = df_raw.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        gdf = gdf.dropna(how="all").reset_index(drop=True)
        if gdf.empty:
            continue

        cols = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", 
                "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        gdf = gdf.iloc[:, :len(cols)]
        gdf.columns = cols[:gdf.shape[1]]

        gdf["SUBJECT"] = gdf["SUBJECT"].astype(str).str.strip()
        gdf = gdf[gdf["SUBJECT"].str.len() > 2]

        for col in gdf.columns:
            if col != "SUBJECT":
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)

        level_cols = [f"LEVEL {i}" for i in range(1, 8) if f"LEVEL {i}" in gdf.columns]
        if gdf.get("TOTAL", pd.Series([0])).sum() == 0:
            gdf["TOTAL"] = gdf[level_cols].sum(axis=1)

        grade_dfs[grade] = gdf
    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== HELPERS ======================
def save_fig_to_bytes(fig):
    img = BytesIO()
    fig.savefig(img, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    img.seek(0)
    return img

# ====================== PROFESSIONAL WORD REPORT ======================
def generate_professional_report(grade_dfs, term):
    doc = Document()
    doc.add_heading(f"Saul Damon High School\n{term} Performance Report", 0).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Executive Summary", level=1)
    total_learners = sum(int(gdf["TOTAL"].sum()) for gdf in grade_dfs.values() if not gdf.empty)
    doc.add_paragraph(f"Total Learners: {total_learners}")

    for grade, gdf in grade_dfs.items():
        if gdf.empty: continue
        doc.add_page_break()
        doc.add_heading(grade, level=1)
        
        avg = gdf["AVERAGE MARK"].mean()
        doc.add_paragraph(f"Average Mark: {avg:.1f}% | Total Learners: {int(gdf['TOTAL'].sum())}")

        # Bar chart in report
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="AVERAGE MARK", y="SUBJECT", data=gdf.sort_values("AVERAGE MARK", ascending=False), 
                    palette="Blues_d", ax=ax)
        ax.set_title(f"{grade} Subject Performance")
        doc.add_picture(save_fig_to_bytes(fig), width=Inches(6.5))

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# ====================== DOWNLOAD FULL REPORT ======================
st.download_button(
    "📄 Download Full Professional Word Report (with Charts)",
    data=generate_professional_report(grade_dfs, term),
    file_name=f"{term}_Full_School_Report.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    use_container_width=True
)

# ====================== DASHBOARD ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty: continue

    st.markdown(f'<p class="section-header">{grade}</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", len(gdf))
    c2.metric("Average Mark", f"{gdf['AVERAGE MARK'].mean():.1f}%")
    c3.metric("Learners", int(gdf["TOTAL"].sum()))

    # Main Chart
    st.subheader("Subject Performance")
    if chart_type == "Bar":
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x="AVERAGE MARK", y="SUBJECT", data=gdf.sort_values("AVERAGE MARK", ascending=False), palette="Blues_d", ax=ax)
        st.pyplot(fig)
    elif chart_type == "Pie":
        fig, ax = plt.subplots(figsize=(8,8))
        ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        level_cols = [c for c in gdf.columns if "LEVEL" in c]
        fig, ax = plt.subplots(figsize=(10,6))
        gdf.set_index("SUBJECT")[level_cols].plot(kind="barh", stacked=True, ax=ax, colormap="Blues")
        st.pyplot(fig)

    # SAFE Level Distribution (This was causing the error)
    st.subheader("Level Distribution per Subject")
    level_cols = [f"LEVEL {i}" for i in range(1, 8) if f"LEVEL {i}" in gdf.columns]
    cols = st.columns(3)

    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 3]:
            # Safe extraction
            levels = np.array([float(row.get(col, 0)) for col in level_cols])
            levels = np.nan_to_num(levels, nan=0.0)  # Force clean values
            
            if levels.sum() <= 0:
                st.write(f"**{row['SUBJECT']}**: No data")
                continue

            # Remove zeros
            positive_mask = levels > 0
            valid_levels = levels[positive_mask]
            valid_labels = [level_cols[j] for j in range(len(levels)) if positive_mask[j]]

            if len(valid_levels) == 0:
                st.write(f"**{row['SUBJECT']}**: No data")
                continue

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(valid_levels, labels=valid_labels, autopct='%1.1f%%', startangle=90,
                   colors=sns.color_palette("Blues", len(valid_levels)))
            ax.set_title(row["SUBJECT"])
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig)

    # Insights
    st.subheader("🔍 Insights & Recommendations")
    for _, row in gdf.iterrows():
        total = row["TOTAL"]
        if total <= 0: continue
        fail_rate = (row.get("LEVEL 1", 0) / total) * 100
        avg_mark = row["AVERAGE MARK"]
        
        if fail_rate > 30:
            st.error(f"**{row['SUBJECT']}**: High failure rate ({fail_rate:.1f}%) — Recommend urgent intervention")
        elif avg_mark < 50:
            st.warning(f"**{row['SUBJECT']}**: Low average ({avg_mark:.1f}%)")
        else:
            st.success(f"**{row['SUBJECT']}**: Strong performance")

st.caption("Saul Damon High School • Professional Term Analysis Dashboard")
