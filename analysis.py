import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from io import BytesIO
import numpy as np

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Saul Damon High School | Term Analysis",
    layout="wide",
    page_icon="📊"
)

# ====================== STYLING ======================
st.markdown("""
<style>
    .stApp {background-color:#F5F7FA; color:#1F2937; font-family:Arial;}
    .main-title {font-size:42px; text-align:center; font-weight:bold; color:#1E3A8A;}
    .sub-title {font-size:24px; text-align:center; margin-bottom:30px; color:#4B5563;}
    .section-header {font-size:24px; font-weight:700; color:#1E40AF; margin:25px 0 10px 0;}
    .insight-box {padding:15px; border-radius:8px; margin:10px 0;}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📋 Settings")
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE (Excel)", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"])
    chart_type = st.selectbox("Main Chart Type", ["Bar", "Stacked Bar", "Pie"])

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis Dashboard</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👆 Please upload your Excel TERM TEMPLATE to begin analysis.")
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
            
        cols = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3",
                "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        
        gdf = gdf.iloc[:, :len(cols)]
        gdf.columns = cols[:gdf.shape[1]]
        
        gdf["SUBJECT"] = gdf["SUBJECT"].astype(str).str.strip()
        gdf = gdf[gdf["SUBJECT"].str.len() > 2]
        
        numeric_cols = ["AVERAGE MARK"] + [f"LEVEL {i}" for i in range(1, 8)]
        for col in numeric_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce").fillna(0)
        
        level_cols = [f"LEVEL {i}" for i in range(1, 8) if f"LEVEL {i}" in gdf.columns]
        if "TOTAL" not in gdf.columns or gdf["TOTAL"].sum() == 0:
            gdf["TOTAL"] = gdf[level_cols].sum(axis=1)
        else:
            gdf["TOTAL"] = pd.to_numeric(gdf["TOTAL"], errors="coerce").fillna(0)
        
        grade_dfs[grade] = gdf
    
    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== CHART HELPERS ======================
def create_bar_chart(gdf, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="AVERAGE MARK", y="SUBJECT", data=gdf.sort_values("AVERAGE MARK", ascending=False), 
                palette="Blues_d", ax=ax)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Average Mark (%)")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f')
    plt.tight_layout()
    return fig

def create_pie_chart(gdf, title, values_col="AVERAGE MARK"):
    fig, ax = plt.subplots(figsize=(8, 8))
    values = gdf[values_col]
    labels = gdf["SUBJECT"]
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
           colors=sns.color_palette("pastel", len(labels)))
    ax.set_title(title)
    ax.axis('equal')
    return fig

def save_fig_to_bytes(fig):
    img = BytesIO()
    fig.savefig(img, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    img.seek(0)
    return img

# ====================== PROFESSIONAL WORD REPORT ======================
def generate_professional_report(grade_dfs, term):
    doc = Document()
    title = doc.add_heading(f"Saul Damon High School\n{term} Performance Report", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_heading("Executive Summary", level=1)
    total_learners = 0
    overall_avg = 0
    total_subjects = 0
    
    for gdf in grade_dfs.values():
        if not gdf.empty:
            total_learners += int(gdf["TOTAL"].sum())
            overall_avg += gdf["AVERAGE MARK"].mean() * len(gdf)
            total_subjects += len(gdf)
    
    if total_subjects > 0:
        overall_avg = overall_avg / total_subjects
    
    doc.add_paragraph(f"Total Learners: {total_learners}")
    doc.add_paragraph(f"Overall Average Mark: {overall_avg:.1f}%")
    doc.add_paragraph("Detailed subject and level analysis with recommendations follow.")
    
    for grade, gdf in grade_dfs.items():
        if gdf.empty:
            continue
        doc.add_page_break()
        doc.add_heading(grade, level=1)
        
        doc.add_paragraph(f"Average Mark: {gdf['AVERAGE MARK'].mean():.1f}%")
        doc.add_paragraph(f"Total Learners: {int(gdf['TOTAL'].sum())}")
        
        fig = create_bar_chart(gdf, f"{grade} - Subject Average Marks")
        doc.add_picture(save_fig_to_bytes(fig), width=Inches(6.5))
        
        # Table
        doc.add_heading("Subject Performance Table", level=2)
        table = doc.add_table(rows=1, cols=4)
        table.style = 'Table Grid'
        hdr = table.rows[0].cells
        for i, h in enumerate(["Subject", "Avg Mark", "Learners", "Pass Rate"]):
            hdr[i].text = h
        
        for _, row in gdf.iterrows():
            cells = table.add_row().cells
            cells[0].text = str(row["SUBJECT"])
            cells[1].text = f"{row['AVERAGE MARK']:.1f}%"
            cells[2].text = str(int(row["TOTAL"]))
            pass_rate = ((row["TOTAL"] - row.get("LEVEL 1", 0)) / row["TOTAL"] * 100) if row["TOTAL"] > 0 else 0
            cells[3].text = f"{pass_rate:.1f}%"
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# ====================== DOWNLOAD FULL REPORT ======================
full_report = generate_professional_report(grade_dfs, term)
st.download_button(
    "📄 Download Full Professional School Report (with Charts)",
    data=full_report,
    file_name=f"Saul_Damon_{term.replace(' ', '_')}_Full_Report.docx",
    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    use_container_width=True
)

# ====================== DASHBOARD ======================
st.markdown("---")

for grade, gdf in grade_dfs.items():
    if gdf.empty:
        continue
        
    st.markdown(f'<p class="section-header">{grade}</p>', unsafe_allow_html=True)
    
    grade_report = generate_professional_report({grade: gdf}, term)
    st.download_button(
        f"📥 Download {grade} Report", 
        data=grade_report,
        file_name=f"{grade}_{term.replace(' ', '_')}_Report.docx"
    )
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects", len(gdf))
    c2.metric("Avg Mark", f"{gdf['AVERAGE MARK'].mean():.1f}%")
    c3.metric("Learners", int(gdf["TOTAL"].sum()))
    pass_rate = ((gdf["TOTAL"].sum() - gdf.get("LEVEL 1", pd.Series(0)).sum()) / gdf["TOTAL"].sum() * 100) if gdf["TOTAL"].sum() > 0 else 0
    c4.metric("Pass Rate", f"{pass_rate:.1f}%")
    
    # Main Chart
    st.subheader("Subject Performance")
    if chart_type == "Bar":
        fig = create_bar_chart(gdf, f"{grade} Subject Averages")
        st.pyplot(fig)
    elif chart_type == "Pie":
        fig = create_pie_chart(gdf, f"{grade} Average Marks")
        st.pyplot(fig)
    else:
        level_cols = [f"LEVEL {i}" for i in range(1, 8) if f"LEVEL {i}" in gdf.columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        gdf.set_index("SUBJECT")[level_cols].plot(kind="barh", stacked=True, ax=ax, colormap="Blues")
        ax.set_title(f"{grade} Level Distribution")
        st.pyplot(fig)
    
    st.dataframe(gdf.round(1), use_container_width=True)
    
    # FIXED: Level Distribution Pies
    st.markdown('<p class="section-header">Level Distribution per Subject</p>', unsafe_allow_html=True)
    level_cols = [f"LEVEL {i}" for i in range(1, 8) if f"LEVEL {i}" in gdf.columns]
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 3]:
            levels = row[level_cols].astype(float)
            total = levels.sum()
            
            if total <= 0:
                st.write(f"{row['SUBJECT']}: No data")
                continue
                
            # Filter out zero values for pie chart
            valid_mask = levels > 0
            valid_levels = levels[valid_mask]
            valid_labels = [col for col, val in zip(level_cols, levels) if val > 0]
            
            if len(valid_levels) == 0:
                st.write(f"{row['SUBJECT']}: No data")
                continue
                
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(valid_levels, labels=valid_labels, autopct='%1.1f%%', 
                   startangle=90, colors=sns.color_palette("Blues", len(valid_levels)))
            ax.set_title(row["SUBJECT"], fontsize=11)
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig)
    
    # Pass / Fail
    st.markdown('<p class="section-header">Pass / Fail</p>', unsafe_allow_html=True)
    gdf = gdf.copy()
    gdf["FAILED"] = gdf.get("LEVEL 1", 0)
    gdf["PASSED"] = gdf["TOTAL"] - gdf["FAILED"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    gdf.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="barh", stacked=True, ax=ax,
                                                        color=["#EF4444", "#10B981"])
    ax.set_title(f"{grade} Pass/Fail by Subject")
    st.pyplot(fig)
    
    # Insights
    st.markdown('<p class="section-header">Insights & Recommendations</p>', unsafe_allow_html=True)
    for _, row in gdf.iterrows():
        if row["TOTAL"] == 0:
            continue
        fail_rate = (row["FAILED"] / row["TOTAL"]) * 100
        avg = row["AVERAGE MARK"]
        
        if fail_rate > 35:
            st.error(f"**{row['SUBJECT']}**: High failure rate ({fail_rate:.1f}%). Urgent intervention needed.")
        elif fail_rate > 20 or avg < 55:
            st.warning(f"**{row['SUBJECT']}**: Needs attention (Avg: {avg:.1f}%, Fail: {fail_rate:.1f}%)")
        else:
            st.success(f"**{row['SUBJECT']}**: Strong performance")

st.caption("Saul Damon High School • Professional Analysis Dashboard")
