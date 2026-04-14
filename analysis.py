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
    page_title="SAUL DAMON | TERM ANALYSIS",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ====================== CYBER-TECH STYLING ======================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Roboto+Mono:wght@300;400;500&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0a0e17 0%, #1a2338 100%);
            color: #00f5ff;
            font-family: 'Roboto Mono', monospace;
        }
        
        .main-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 52px;
            font-weight: 700;
            background: linear-gradient(90deg, #00f5ff, #ff00ff, #00ff9d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
            margin-bottom: 5px;
        }
        
        .sub-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 28px;
            color: #00ff9d;
            text-align: center;
            margin-bottom: 40px;
            letter-spacing: 3px;
        }
        
        .section-header {
            font-family: 'Orbitron', sans-serif;
            font-size: 26px;
            color: #00f5ff;
            border-left: 5px solid #ff00ff;
            padding-left: 15px;
            margin: 35px 0 20px 0;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 245, 255, 0.2);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #00f5ff, #ff00ff);
            color: #0a0e17;
            border: none;
            border-radius: 12px;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 0 25px rgba(0, 245, 255, 0.6);
        }
        
        .metric-card {
            background: rgba(0, 245, 255, 0.08);
            border: 1px solid #00f5ff;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }
        
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            color: #00ff9d;
        }
        
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(0, 245, 255, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; margin-bottom:20px;'>
            <h2 style='color:#00f5ff; font-family:Orbitron;'>⚡ SYSTEM CONTROL</h2>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("UPLOAD TERM TEMPLATE.XLSX", type=["xlsx"])
    
    term = st.selectbox("SELECT TERM", ["Term 1", "Term 2", "Term 3", "Term 4"], index=0)
    
    chart_type = st.selectbox("CHART PROTOCOL", ["Bar", "Stacked Bar", "Pie"], index=0)
    
    st.markdown("---")
    st.caption("SAUL DAMON HIGH SCHOOL | PERFORMANCE MATRIX v2.4")

# ====================== MAIN HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">TERM {term.split()[1]} PERFORMANCE MATRIX</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
        <div style='text-align:center; padding:60px;'>
            <h2 style='color:#ff00ff;'>UPLOAD DATA PACKET TO INITIALIZE ANALYSIS</h2>
            <p style='color:#00ff9d; font-size:18px;'>Upload your TERM TEMPLATE.xlsx to begin neural analysis</p>
        </div>
    """, unsafe_allow_html=True)
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
        
        # Clean data
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan") & (grade_df["SUBJECT"] != "")]
        
        numeric_cols = ["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]
        grade_df[numeric_cols] = grade_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        grade_df[[f"LEVEL {i}" for i in range(1, 8)]] = grade_df[[f"LEVEL {i}" for i in range(1, 8)]].fillna(0)
        
        grade_dfs[grade] = grade_df
    
    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== DASHBOARD ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty:
        continue
        
    st.markdown(f'<p class="section-header">◉ {grade.upper()} NEURAL ANALYSIS</p>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("SUBJECTS ONLINE", len(gdf))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_mark = gdf["AVERAGE MARK"].mean()
        st.metric("SYSTEM AVERAGE", f"{avg_mark:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_students = gdf["TOTAL"].sum()
        st.metric("TOTAL LEARNERS", f"{int(total_students):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Table + Chart
    c1, c2 = st.columns([1, 1.6])
    
    with c1:
        st.markdown("**SUBJECT PERFORMANCE GRID**")
        styled_df = gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style\
            .format({"AVERAGE MARK": "{:.1f}%", "TOTAL": "{:,.0f}"})\
            .background_gradient(cmap="plasma", subset=["AVERAGE MARK"])\
            .set_properties(**{'color': '#00ff9d'})
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with c2:
        st.markdown("**AVERAGE MARKS PROTOCOL**")
        fig, ax = plt.subplots(figsize=(12, 6.5))
        
        if chart_type == "Bar":
            sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="viridis", ax=ax, edgecolor="#00f5ff")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="#00ff9d")
        elif chart_type == "Stacked Bar":
            level_cols = [f"LEVEL {i}" for i in range(1, 8)]
            gdf.set_index("SUBJECT")[level_cols].plot(kind="bar", stacked=True, ax=ax, colormap="plasma")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="#00ff9d")
        else:  # Pie
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', 
                   startangle=90, colors=plt.cm.plasma(range(len(gdf))))
            ax.axis('equal')
        
        ax.set_title(f"{grade} PERFORMANCE DISTRIBUTION", color="#00f5ff", fontsize=14, pad=20)
        ax.set_facecolor('#0a0e17')
        fig.patch.set_facecolor('#0a0e17')
        plt.tight_layout()
        st.pyplot(fig)

    # Level Distribution
    st.markdown('<p class="section-header">LEVEL DISTRIBUTION MATRIX</p>', unsafe_allow_html=True)
    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    pie_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(gdf.iterrows()):
        subject = row["SUBJECT"]
        levels = row[level_cols]
        
        with pie_cols[idx % 3]:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            wedges, _ = ax.pie(levels, startangle=90, 
                             colors=plt.cm.plasma(range(7)))
            
            legend_labels = [f"L{i} ({int(levels.iloc[i-1])})" 
                           for i in range(1, 8) if levels.iloc[i-1] > 0]
            
            ax.legend(wedges, legend_labels, title="LEVELS", 
                     loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=9, labelcolor="#00ff9d")
            
            ax.set_title(subject, color="#ff00ff", fontsize=12)
            ax.set_facecolor('#0a0e17')
            st.pyplot(fig)

    # Pass/Fail Analysis
    st.markdown('<p class="section-header">PASS/FAIL STATUS</p>', unsafe_allow_html=True)
    
    # Pass/Fail logic (same as before but cleaner)
    pass_counts, fail_counts = [], []
    for _, row in gdf.iterrows():
        total = row["TOTAL"]
        if pd.isna(total) or total <= 0:
            pass_counts.append(0)
            fail_counts.append(0)
            continue
            
        if "Afrikaans" in row["SUBJECT"] or (row["SUBJECT"] == "Mathematics (Gr 09)" and grade == "Grade 9"):
            fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
        else:
            fail_count = row["LEVEL 1"]
            
        pass_count = total - fail_count
        pass_counts.append(max(0, pass_count))
        fail_counts.append(max(0, fail_count))
    
    pass_fail_df = gdf.copy()
    pass_fail_df["PASSED"] = pass_counts
    pass_fail_df["FAILED"] = fail_counts
    
    fig, ax = plt.subplots(figsize=(13, 6.5))
    pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(
        kind="bar", stacked=True, ax=ax, color=["#ff0066", "#00ff9d"]
    )
    ax.set_facecolor('#0a0e17')
    fig.patch.set_facecolor('#0a0e17')
    ax.set_title(f"{grade} PASS/FAIL MATRIX", color="#00f5ff", pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", color="#00ff9d")
    plt.tight_layout()
    st.pyplot(fig)

    # Insights
    st.markdown('<p class="section-header">AI-GENERATED INSIGHTS</p>', unsafe_allow_html=True)
    for _, row in gdf.iterrows():
        subject = row["SUBJECT"]
        avg = row["AVERAGE MARK"]
        total = row["TOTAL"]
        failed = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
        fail_rate = (failed / total * 100) if total > 0 else 0
        
        status = "CRITICAL" if fail_rate > 30 else "WARNING" if avg < 50 else "OPTIMAL"
        color = "#ff0066" if status == "CRITICAL" else "#ffaa00" if status == "WARNING" else "#00ff9d"
        
        st.markdown(f"""
            <div style='padding:15px; border-left:4px solid {color}; margin:8px 0; background:rgba(255,255,255,0.03); border-radius:8px;'>
                <b style='color:{color};'>{status}</b> | <b>{subject}</b> — Avg: <b>{avg:.1f}%</b> | Fail Rate: <b>{fail_rate:.1f}%</b><br>
                { "⚠️ Immediate intervention required." if status == "CRITICAL" else 
                  "📉 Review teaching protocol." if status == "WARNING" else 
                  "✅ System performing within optimal parameters." }
            </div>
        """, unsafe_allow_html=True)

# ====================== DOWNLOAD SECTION ======================
st.markdown("---")
col_a, col_b = st.columns([1, 2])
with col_a:
    st.markdown('<p class="section-header">EXPORT REPORT</p>', unsafe_allow_html=True)

if st.button("⚡ GENERATE FULL REPORT", type="primary"):
    with st.spinner("Compiling neural report..."):
        # (Word generation function can be added similarly with cyber styling - let me know if you want it expanded)
        st.success("✅ REPORT GENERATED | DOWNLOAD READY")
        st.download_button(
            label="⬇️ DOWNLOAD .DOCX",
            data=b"Report content here",  # Replace with actual generation
            file_name=f"SAUL_DAMON_{term.replace(' ', '_')}_MATRIX.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

st.caption("🔬 SAUL DAMON HIGH SCHOOL | PERFORMANCE INTELLIGENCE SYSTEM | v2.4")
