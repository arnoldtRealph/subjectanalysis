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
    .stApp {background-color:#F8FAFC; color:#1F2937;}
    .main-title {font-size:48px; text-align:center; font-weight:bold; color:#1E3A8A; margin-bottom:10px;}
    .sub-title {font-size:26px; text-align:center; margin-bottom:40px; color:#334155;}
    .section-header {font-size:28px; font-weight:700; color:#1E40AF; margin:35px 0 15px 0;}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE (Excel)", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"])
    chart_type = st.selectbox("Main Chart Type", ["Bar", "Stacked Bar", "Pie"])

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis Dashboard</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👆 Upload your Excel TERM TEMPLATE to begin")
    st.stop()

# ====================== DATA PROCESSING ======================
@st.cache_data
def process_data(file):
    df_raw = pd.read_excel(file, header=None)
    grade_starts = df_raw.index[df_raw[0].astype(str).str.contains("GRADE", na=False, case=False)].tolist()
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_dfs = {}

    for i, grade in enumerate(grades):
        if i >= len(grade_starts): break
        start_idx = grade_starts[i] + 2
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df_raw)
        
        gdf = df_raw.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        gdf = gdf.dropna(how="all").reset_index(drop=True)
        if gdf.empty: continue

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
    fig.savefig(img, format='png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    img.seek(0)
    return img

# ====================== PROFESSIONAL WORD REPORT (with Pie Charts) ======================
def generate_professional_report(grade_dfs, term):
    doc = Document()
    doc.add_heading(f"Saul Damon High School\n{term} Performance Report", 0).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Executive Summary", level=1)
    for grade, gdf in grade_dfs.items():
        if gdf.empty: continue
        avg = gdf["AVERAGE MARK"].mean()
        doc.add_paragraph(f"{grade} Average: {avg:.1f}%")

    for grade, gdf in grade_dfs.items():
        if gdf.empty: continue
        doc.add_page_break()
        doc.add_heading(grade, level=1)

        # Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="AVERAGE MARK", y="SUBJECT", data=gdf.sort_values("AVERAGE MARK", ascending=False), 
                    palette="Blues_d", ax=ax)
        ax.set_title(f"{grade} - Subject Average Marks")
        doc.add_picture(save_fig_to_bytes(fig), width=Inches(6.5))

        # Pie Chart per Subject (in report)
        doc.add_heading("Level Distribution", level=2)
        level_cols = [c for c in gdf.columns if "LEVEL" in c]
        for _, row in gdf.iterrows():
            levels = np.array([float(row.get(col, 0)) for col in level_cols])
            levels = np.nan_to_num(levels, nan=0.0)
            if levels.sum() <= 0: continue

            positive_mask = levels > 0.01
            valid_levels = levels[positive_mask]
            valid_labels = [level_cols[j] for j in range(len(levels)) if positive_mask[j]]

            if len(valid_levels) == 0: continue

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(valid_levels, labels=valid_labels, autopct='%1.1f%%', startangle=90,
                   colors=sns.color_palette("Blues", len(valid_levels)))
            ax.set_title(f"{row['SUBJECT']} - Level Distribution")
            ax.axis('equal')
            doc.add_picture(save_fig_to_bytes(fig), width=Inches(5.5))
            plt.close(fig)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

st.download_button(
    "📄 Download Full Professional Word Report (with Bar + Pie Charts)",
    data=generate_professional_report(grade_dfs, term),
    file_name=f"{term}_Full_School_Report.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    use_container_width=True
)

# ====================== BEAUTIFUL DASHBOARD ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty: continue

    st.markdown(f'<p class="section-header">{grade}</p>', unsafe_allow_html=True)

    # Stats only (no total learners)
    col1, col2 = st.columns(2)
    col1.metric("Number of Subjects", len(gdf))
    col2.metric("Average Mark", f"{gdf['AVERAGE MARK'].mean():.1f}%")

    st.markdown("### Average Marks by Subject")
    if chart_type == "Bar" or chart_type == "Stacked Bar":
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.barplot(x="AVERAGE MARK", y="SUBJECT", data=gdf.sort_values("AVERAGE MARK", ascending=False), 
                    palette="Blues_d", ax=ax)
        ax.set_xlabel("Average Mark (%)")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # Per Subject Pie Charts
    st.markdown("### Level Distribution per Subject")
    level_cols = [f"LEVEL {i}" for i in range(1, 8) if f"LEVEL {i}" in gdf.columns]
    cols = st.columns(3)

    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 3]:
            levels = np.array([float(row.get(col, 0)) for col in level_cols])
            levels = np.nan_to_num(levels, nan=0.0)
            
            if levels.sum() <= 0:
                st.write(f"**{row['SUBJECT']}**: No data")
                continue

            positive_mask = levels > 0.01
            valid_levels = levels[positive_mask]
            valid_labels = [level_cols[j] for j in range(len(levels)) if positive_mask[j]]

            if len(valid_levels) == 0:
                st.write(f"**{row['SUBJECT']}**: No data")
                continue

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(valid_levels, labels=valid_labels, autopct='%1.1f%%', startangle=90,
                   colors=sns.color_palette("Blues", len(valid_levels)))
            ax.set_title(row["SUBJECT"], fontsize=12)
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig)

    # Insights
    st.markdown("### 🔍 Insights & Recommendations")
    for _, row in gdf.iterrows():
        total = row["TOTAL"]
        if total <= 0: continue
        fail_rate = (row.get("LEVEL 1", 0) / total) * 100
        if fail_rate > 30:
            st.error(f"**{row['SUBJECT']}**: High failure rate ({fail_rate:.1f}%) — Immediate intervention recommended")
        elif row["AVERAGE MARK"] < 52:
            st.warning(f"**{row['SUBJECT']}**: Below target average ({row['AVERAGE MARK']:.1f}%)")
        else:
            st.success(f"**{row['SUBJECT']}**: Strong performance")

st.caption("Saul Damon High School • Professional Term Analysis")
