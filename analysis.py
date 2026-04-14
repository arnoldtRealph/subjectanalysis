import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re

# ====================== STREAMLIT CONFIG ======================
st.set_page_config(
    page_title="Saul Damon High School | Term Analysis",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ====================== PROFESSIONAL TECH STYLING ======================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0A2540 0%, #112B4F 100%);
            color: #E0F2FF;
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 48px;
            font-weight: 700;
            background: linear-gradient(90deg, #00D4C8, #4FC3F7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 8px;
        }
        
        .sub-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 26px;
            color: #81E6D9;
            text-align: center;
            margin-bottom: 40px;
            letter-spacing: 1.5px;
        }
        
        .section-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 24px;
            color: #00D4C8;
            border-bottom: 2px solid #00D4C8;
            padding-bottom: 10px;
            margin: 35px 0 20px 0;
            letter-spacing: 1px;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 212, 200, 0.25);
            border-radius: 16px;
            padding: 20px;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #00D4C8, #4FC3F7);
            color: #0A2540;
            border: none;
            border-radius: 12px;
            padding: 12px 32px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 212, 200, 0.4);
        }
        
        .metric-card {
            background: rgba(0, 212, 200, 0.08);
            border: 1px solid #00D4C8;
            border-radius: 14px;
            padding: 18px;
            text-align: center;
        }
        
        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: #81E6D9;
        }
        
        .stDataFrame {
            border-radius: 12px;
            border: 1px solid rgba(0, 212, 200, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; margin-bottom:25px;'>
            <h2 style='color:#00D4C8; font-family:Space Grotesk;'>TERM PERFORMANCE DASHBOARD</h2>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE Excel File", type=["xlsx"])
    
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"], index=0)
    
    chart_type = st.selectbox("Chart Type", ["Bar", "Stacked Bar", "Pie"], index=0)

# ====================== MAIN HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} • Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👈 Upload your TERM TEMPLATE.xlsx file from the sidebar to generate the analysis.")
    st.stop()

# ====================== DATA PROCESSING ======================
@st.cache_data
def process_data(file):
    df = pd.read_excel(file, header=None)
    
    grade_starts = df.index[df[0].str.contains("GRADE", na=False)].tolist()
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_dfs = {}
    
    for i, grade in enumerate(grades):
        if i >= len(grade_starts):
            break
        start_idx = grade_starts[i] + 2
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df)
        
        grade_df = df.iloc[start_idx:end_idx].dropna(how="all").reset_index(drop=True)
        
        cols = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", 
                "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        grade_df = grade_df.iloc[:, :len(cols)]
        grade_df.columns = cols
        
        # Clean SUBJECT
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan") & (grade_df["SUBJECT"] != "")]
        
        # Convert numbers
        numeric_cols = ["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]
        grade_df[numeric_cols] = grade_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        grade_df[[f"LEVEL {i}" for i in range(1, 8)]] = grade_df[[f"LEVEL {i}" for i in range(1, 8)]].fillna(0)
        
        grade_dfs[grade] = grade_df
    
    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== DASHBOARD CONTENT ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty:
        continue
        
    st.markdown(f'<p class="section-header">{grade} Analysis</p>', unsafe_allow_html=True)
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Subjects", len(gdf))
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_mark = gdf["AVERAGE MARK"].mean()
        st.metric("Average Mark", f"{avg_mark:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_learners = gdf["TOTAL"].sum()
        st.metric("Total Learners", f"{int(total_learners):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Table and Main Chart
    c1, c2 = st.columns([1, 1.7])
    with c1:
        st.subheader("Subject Performance")
        styled = gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style\
            .format({"AVERAGE MARK": "{:.1f}%", "TOTAL": "{:,.0f}"})\
            .background_gradient(cmap="Blues", subset=["AVERAGE MARK"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    
    with c2:
        st.subheader("Average Marks Visualization")
        fig, ax = plt.subplots(figsize=(12, 6.5))
        
        if chart_type == "Bar":
            sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues_d", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        elif chart_type == "Stacked Bar":
            level_cols = [f"LEVEL {i}" for i in range(1, 8)]
            gdf.set_index("SUBJECT")[level_cols].plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        else:  # Pie
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', 
                   startangle=90, colors=sns.color_palette("Blues_d", len(gdf)))
            ax.axis('equal')
        
        ax.set_title(f"{grade} - Average Marks", fontsize=14, color="#00D4C8", pad=20)
        plt.tight_layout()
        st.pyplot(fig)

    # Level Distribution
    st.markdown('<p class="section-header">Level Distribution per Subject</p>', unsafe_allow_html=True)
    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    pie_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(gdf.iterrows()):
        subject = row["SUBJECT"]
        levels = row[level_cols]
        
        with pie_cols[idx % 3]:
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            wedges, _ = ax.pie(levels, startangle=90, colors=sns.color_palette("Blues", 7))
            
            legend_labels = [f"Level {i} ({int(levels.iloc[i-1])})" 
                           for i in range(1, 8) if levels.iloc[i-1] > 0]
            
            ax.legend(wedges, legend_labels, title="Levels", 
                     loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=9)
            
            ax.set_title(subject, fontsize=11, color="#81E6D9")
            ax.axis('equal')
            st.pyplot(fig)

    # Pass/Fail
    st.markdown('<p class="section-header">Pass / Fail Distribution</p>', unsafe_allow_html=True)
    
    pass_counts = []
    fail_counts = []
    for _, row in gdf.iterrows():
        total = row["TOTAL"]
        if pd.isna(total) or total <= 0:
            pass_counts.append(0)
            fail_counts.append(0)
            continue
            
        if "Afrikaans HL" in row["SUBJECT"] or "Afrikaans FAL" in row["SUBJECT"] or \
           (row["SUBJECT"] == "Mathematics (Gr 09)" and grade == "Grade 9"):
            fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
        else:
            fail_count = row.get("LEVEL 1", 0)
            
        pass_count = total - fail_count
        pass_counts.append(max(0, pass_count))
        fail_counts.append(max(0, fail_count))
    
    pass_fail_df = gdf.copy()
    pass_fail_df["PASSED"] = pass_counts
    pass_fail_df["FAILED"] = fail_counts
    
    fig, ax = plt.subplots(figsize=(13, 6.5))
    pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(
        kind="bar", stacked=True, ax=ax, color=["#FF6B6B", "#4ECDC4"]
    )
    ax.set_title(f"{grade} - Pass/Fail Distribution", fontsize=14, color="#00D4C8", pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Insights
    st.markdown('<p class="section-header">Insights & Recommendations</p>', unsafe_allow_html=True)
    for _, row in gdf.iterrows():
        subject = row["SUBJECT"]
        avg = row["AVERAGE MARK"]
        total = row["TOTAL"]
        failed = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
        fail_rate = (failed / total * 100) if total > 0 else 0
        pass_rate = 100 - fail_rate
        
        if fail_rate > 30:
            insight = f"High fail rate detected. Consider targeted intervention."
            emoji = "🔴"
        elif avg < 50:
            insight = f"Below target average. Review teaching methods and resources."
            emoji = "🟠"
        else:
            insight = f"Strong performance. Continue current strategies."
            emoji = "🟢"
        
        st.markdown(f"""
            **{subject}** — Avg: **{avg:.1f}%** | Pass Rate: **{pass_rate:.1f}%**  
            {emoji} {insight}
        """)

# ====================== DOWNLOAD SECTION ======================
st.markdown("---")
st.markdown('<p class="section-header">Generate Full Report</p>', unsafe_allow_html=True)

if st.button("Generate Professional Word Report", type="primary"):
    with st.spinner("Generating report..."):
        st.success("Report ready! (Word generation logic can be expanded here)")
        # Add your full generate_word_report() function here if needed

st.caption("Saul Damon High School • Professional Term Performance Dashboard")
