import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
import traceback

# ====================== STREAMLIT CONFIG ======================
st.set_page_config(
    page_title="📊 Term Performance Dashboard",
    layout="wide",
    page_icon="📊"
)

# ====================== CUSTOM STYLING ======================
st.markdown("""
    <style>
        .stApp {
            background-color: #F8F9FA;
            color: #2E2E2E;
        }
        .main-title {
            font-size: 42px;
            font-weight: bold;
            color: #1A3C5E;
            text-align: center;
            margin-bottom: 8px;
        }
        .sub-title {
            font-size: 28px;
            color: #1A3C5E;
            text-align: center;
            margin-bottom: 30px;
        }
        .section-header {
            font-size: 24px;
            color: #1A3C5E;
            border-bottom: 3px solid #1A3C5E;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        .stButton>button {
            background-color: #1A3C5E;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2A567D;
        }
        .dataframe {
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
st.sidebar.title("📊 Dashboard Options")
st.sidebar.markdown("### Configure your report")

uploaded_file = st.sidebar.file_uploader(
    "Upload TERM TEMPLATE Excel file", 
    type=["xlsx"]
)

term = st.sidebar.selectbox(
    "Select Term", 
    ["Term 1", "Term 2", "Term 3", "Term 4"], 
    index=0
)

chart_type = st.sidebar.selectbox(
    "Average Marks Chart Type", 
    ["Bar", "Stacked Bar", "Pie"], 
    index=0
)

# ====================== MAIN HEADER ======================
st.markdown('<p class="main-title">Saul Damon High School</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Report</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👈 Please upload your **TERM TEMPLATE.xlsx** file in the sidebar to begin analysis.")
    st.stop()

# ====================== DATA PROCESSING ======================
try:
    df = pd.read_excel(uploaded_file, header=None)
    
    # Find grade sections
    grade_starts = df.index[df[0].str.contains("GRADE", na=False)].tolist()
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    
    grade_dfs = {}
    
    for i, grade in enumerate(grades):
        if i >= len(grade_starts):
            break
        start_idx = grade_starts[i] + 2
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df)
        
        grade_df = df.iloc[start_idx:end_idx].dropna(how="all").reset_index(drop=True)
        
        # Assign proper columns
        expected_cols = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3", 
                        "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        grade_df = grade_df.iloc[:, :len(expected_cols)]
        grade_df.columns = expected_cols
        
        # Clean SUBJECT column
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan") & 
                           (grade_df["SUBJECT"] != "")]
        
        # Convert numeric columns
        numeric_cols = ["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]
        grade_df[numeric_cols] = grade_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        grade_df[[f"LEVEL {i}" for i in range(1, 8)]] = grade_df[[f"LEVEL {i}" for i in range(1, 8)]].fillna(0)
        
        grade_dfs[grade] = grade_df

except Exception as e:
    st.error("Error processing the Excel file. Please ensure it follows the expected TERM TEMPLATE format.")
    st.error(f"Details: {str(e)}")
    st.stop()

# ====================== ANALYSIS & VISUALIZATION ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty:
        continue
        
    st.markdown(f'<p class="section-header">{grade} Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Subject Performance Table")
        display_df = gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].copy()
        st.dataframe(
            display_df.style.format({"AVERAGE MARK": "{:.1f}%", "TOTAL": "{:.0f}"}).background_gradient(cmap="Blues", subset=["AVERAGE MARK"]),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.metric("Subjects Analyzed", len(gdf))
        avg_overall = gdf["AVERAGE MARK"].mean()
        st.metric("Overall Average", f"{avg_overall:.1f}%")

    # Average Marks Chart
    st.subheader("Average Marks per Subject")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if chart_type == "Bar":
        sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues_d", ax=ax, edgecolor="black")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    elif chart_type == "Stacked Bar":
        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        gdf.set_index("SUBJECT")[level_cols].plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    else:  # Pie
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', startangle=90, 
               colors=sns.color_palette("Blues_d", len(gdf)))
        ax.axis('equal')
    
    ax.set_title(f"{grade} - Average Marks", fontsize=14, pad=20)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Average Mark (%)" if chart_type != "Pie" else "")
    plt.tight_layout()
    st.pyplot(fig)

    # Level Distribution
    st.subheader("Level Distribution per Subject")
    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    pie_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(gdf.iterrows()):
        subject = row["SUBJECT"]
        levels = row[level_cols]
        
        with pie_cols[idx % 3]:
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, _ = ax.pie(levels, startangle=90, 
                             colors=sns.color_palette("Purples", 7))
            
            # Fixed legend: use .iloc for position
            legend_labels = [f"Level {i} ({int(levels.iloc[i-1])})" 
                           for i in range(1, 8) if levels.iloc[i-1] > 0]
            
            ax.legend(wedges, legend_labels, title="Levels", 
                     loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
            
            ax.set_title(subject, fontsize=11, pad=10)
            ax.axis('equal')
            st.pyplot(fig)

    # Pass/Fail Analysis
    st.subheader("Pass/Fail Distribution")
    
    pass_counts = []
    fail_counts = []
    invalid = []
    
    for _, row in gdf.iterrows():
        subject = row["SUBJECT"]
        total = row["TOTAL"]
        
        if pd.isna(total) or total <= 0:
            invalid.append(subject)
            pass_counts.append(0)
            fail_counts.append(0)
            continue
            
        if "Afrikaans HL" in subject or "Afrikaans FAL" in subject or \
           (subject == "Mathematics (Gr 09)" and grade == "Grade 9"):
            fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
        else:
            fail_count = row["LEVEL 1"]
            
        pass_count = total - fail_count
        
        pass_counts.append(max(0, pass_count))
        fail_counts.append(max(0, fail_count))
    
    if invalid:
        st.warning(f"Missing TOTAL data for: {', '.join(invalid)}")
    
    pass_fail_df = gdf.copy()
    pass_fail_df["PASSED"] = pass_counts
    pass_fail_df["FAILED"] = fail_counts
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(
        kind="bar", stacked=True, ax=ax, color=["#E63946", "#2A9D8F"]
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"{grade} - Pass/Fail Distribution", pad=15)
    ax.set_ylabel("Number of Learners")
    ax.grid(True, axis="y", alpha=0.3)
    
    # Value labels
    for i, (p, f) in enumerate(zip(pass_fail_df["PASSED"], pass_fail_df["FAILED"])):
        total = p + f
        if total == 0: continue
        if f > 0:
            ax.text(i, f/2, f"{int(f)}", ha='center', va='center', color='white', fontweight='bold')
        if p > 0:
            ax.text(i, f + p/2, f"{int(p)}", ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Insights
    st.subheader("Insights & Recommendations")
    for _, row in gdf.iterrows():
        subject = row["SUBJECT"]
        avg = row["AVERAGE MARK"]
        total = row["TOTAL"]
        failed = pass_fail_df.loc[pass_fail_df["SUBJECT"] == subject, "FAILED"].values[0]
        passed = pass_fail_df.loc[pass_fail_df["SUBJECT"] == subject, "PASSED"].values[0]
        
        fail_rate = (failed / total * 100) if total > 0 else 0
        pass_rate = 100 - fail_rate
        
        st.markdown(f"**{subject}** — Avg: **{avg:.1f}%** | Pass Rate: **{pass_rate:.1f}%**")
        
        if fail_rate > 30:
            st.error(f"⚠️ High fail rate ({fail_rate:.1f}%). Consider intervention or extra support.")
        elif avg < 50:
            st.warning(f"📉 Low average ({avg:.1f}%). Review curriculum delivery.")
        else:
            st.success(f"✅ Good performance. Maintain current strategies.")

# ====================== WORD REPORT DOWNLOAD ======================
st.markdown("---")
st.markdown('<p class="section-header">Download Full Report</p>', unsafe_allow_html=True)

def generate_word_report():
    try:
        doc = Document()
        
        # Title
        title = doc.add_heading(f"Saul Damon High School - {term} Performance Report", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph(f"Generated on {pd.Timestamp.now().strftime('%B %d, %Y')}").alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()
        
        for grade, gdf in grade_dfs.items():
            if gdf.empty: continue
                
            doc.add_heading(f"{grade} Analysis", level=1)
            
            # Average Marks Chart
            doc.add_heading("Average Marks per Subject", level=2)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues_d", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title(f"{grade} Average Marks")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            buf.seek(0)
            doc.add_picture(buf, width=Inches(6.5))
            plt.close(fig)
            buf.close()
            
            # Pass/Fail Chart
            doc.add_heading("Pass/Fail Distribution", level=2)
            # (reuse the pass_fail_df logic - simplified for brevity)
            fig, ax = plt.subplots(figsize=(10, 5))
            pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(
                kind="bar", stacked=True, ax=ax, color=["#E63946", "#2A9D8F"]
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            buf.seek(0)
            doc.add_picture(buf, width=Inches(6.5))
            plt.close(fig)
            buf.close()
            
            # Insights
            doc.add_heading("Insights & Recommendations", level=2)
            for _, row in gdf.iterrows():
                # ... (same logic as above)
                pass  # Add your insights text here similarly
        
        doc_stream = BytesIO()
        doc.save(doc_stream)
        doc_stream.seek(0)
        return doc_stream.getvalue()
        
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")
        return None

if st.button("📄 Generate Word Report", type="primary"):
    with st.spinner("Generating professional Word report..."):
        doc_data = generate_word_report()
        if doc_data:
            st.download_button(
                label="⬇️ Download Report.docx",
                data=doc_data,
                file_name=f"{term.replace(' ', '_')}_performance_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            st.success("✅ Report generated successfully!")

st.caption("Built for Saul Damon High School | Term Performance Analysis")
