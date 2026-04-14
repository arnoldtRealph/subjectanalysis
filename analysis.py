import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Saul Damon High School | Term Analysis",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ====================== STYLING - HIGH READABILITY ======================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0A2540 0%, #112B4F 100%);
            color: #F0FAFF;
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 54px;
            font-weight: 700;
            background: linear-gradient(90deg, #00E6D8, #5EDFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 8px;
        }
        
        .sub-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 29px;
            color: #A5F2E9;
            text-align: center;
            margin-bottom: 40px;
        }
        
        .section-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 27px;
            color: #00E6D8;
            border-bottom: 3px solid #00E6D8;
            padding-bottom: 12px;
            margin: 45px 0 25px 0;
        }
        
        .metric-card {
            background: rgba(0, 230, 216, 0.12);
            border: 1px solid #00E6D8;
            border-radius: 16px;
            padding: 22px;
            text-align: center;
            font-size: 18px;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #00E6D8, #5EDFFF);
            color: #0A2540;
            border-radius: 12px;
            padding: 14px 36px;
            font-weight: 600;
            font-size: 16px;
        }
        
        h1, h2, h3, .stMarkdown p, label, span {
            color: #F0FAFF !important;
        }
        
        .stDataFrame {
            font-size: 15.5px;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("<h2 style='color:#00E6D8; text-align:center;'>TERM PERFORMANCE DASHBOARD</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE Excel File", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"], index=0)
    chart_type = st.selectbox("Chart Type", ["Bar", "Stacked Bar", "Pie"], index=0)

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("👈 Upload your TERM TEMPLATE.xlsx file in the sidebar.")
    st.stop()

# ====================== DATA PROCESSING WITH FIXED TOTALS ======================
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
        
        grade_df = df_raw.iloc[start_idx:end_idx].dropna(how="all").reset_index(drop=True)
        
        cols = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", 
                "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        grade_df = grade_df.iloc[:, :len(cols)]
        grade_df.columns = cols
        
        # Clean subject names
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan") & (grade_df["SUBJECT"] != "")]
        
        # Convert to numeric
        numeric_cols = ["AVERAGE MARK"] + [f"LEVEL {i}" for i in range(1, 8)] + ["TOTAL"]
        grade_df[numeric_cols] = grade_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        
        # FIX: Calculate accurate TOTAL from levels (this solves the mismatch)
        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        grade_df["CALCULATED_TOTAL"] = grade_df[level_cols].sum(axis=1)
        
        # Use calculated total for all calculations
        grade_df["TOTAL"] = grade_df["CALCULATED_TOTAL"]
        
        grade_dfs[grade] = grade_df
    
    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== DASHBOARD ======================
for grade, gdf in grade_dfs.items():
    if gdf.empty:
        continue
        
    st.markdown(f'<p class="section-header">{grade} Analysis</p>', unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Subjects", len(gdf))
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_mark = gdf["AVERAGE MARK"].mean()
        st.metric("Overall Average", f"{avg_mark:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_learners = int(gdf["TOTAL"].sum())
        st.metric("Total Learners", f"{total_learners:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Full-width Table
    st.subheader("Subject Performance Table")
    display_df = gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].copy()
    styled = display_df.style\
        .format({"AVERAGE MARK": "{:.1f}%", "TOTAL": "{:,.0f}"})\
        .background_gradient(cmap="Blues", subset=["AVERAGE MARK"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Full-width Chart
    st.subheader("Average Marks Visualization")
    fig, ax = plt.subplots(figsize=(14, 7.5))
    
    if chart_type == "Bar":
        sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues_d", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    elif chart_type == "Stacked Bar":
        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        gdf.set_index("SUBJECT")[level_cols].plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    else:  # Pie
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', 
               startangle=90, colors=sns.color_palette("Blues_d", len(gdf)))
        ax.axis('equal')
    
    ax.set_title(f"{grade} — Average Marks per Subject", fontsize=16, color="#00E6D8", pad=20)
    plt.tight_layout()
    st.pyplot(fig)

    # Level Distribution (pies)
    st.markdown('<p class="section-header">Level Distribution per Subject</p>', unsafe_allow_html=True)
    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    pie_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(gdf.iterrows()):
        subject = row["SUBJECT"]
        levels = row[level_cols]
        
        with pie_cols[idx % 3]:
            fig, ax = plt.subplots(figsize=(5.8, 5.8))
            wedges, _ = ax.pie(levels, startangle=90, colors=sns.color_palette("Blues", 7))
            
            legend_labels = [f"Level {i} ({int(levels.iloc[i-1])})" 
                           for i in range(1, 8) if levels.iloc[i-1] > 0]
            
            ax.legend(wedges, legend_labels, title="Levels", loc="center left", 
                     bbox_to_anchor=(1.05, 0.5), fontsize=10)
            ax.set_title(subject, fontsize=12, color="#A5F2E9")
            ax.axis('equal')
            st.pyplot(fig)

    # Pass/Fail
    st.markdown('<p class="section-header">Pass / Fail Distribution</p>', unsafe_allow_html=True)
    
    pass_counts, fail_counts = [], []
    for _, row in gdf.iterrows():
        total = row["TOTAL"]
        if pd.isna(total) or total <= 0:
            pass_counts.append(0)
            fail_counts.append(0)
            continue
            
        if any(x in row["SUBJECT"] for x in ["Afrikaans HL", "Afrikaans FAL", "Afrikaans HT"]) or \
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
    
    fig, ax = plt.subplots(figsize=(14, 7.5))
    pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(
        kind="bar", stacked=True, ax=ax, color=["#FF6B6B", "#4ECDC4"]
    )
    ax.set_title(f"{grade} — Pass/Fail Distribution", fontsize=16, color="#00E6D8", pad=20)
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
        
        if fail_rate > 30:
            st.error(f"🔴 **{subject}** — High fail rate ({fail_rate:.1f}%). Targeted support strongly recommended.")
        elif avg < 50:
            st.warning(f"🟠 **{subject}** — Low average ({avg:.1f}%). Review teaching approach.")
        else:
            st.success(f"🟢 **{subject}** — Good performance (Pass rate: {100-fail_rate:.1f}%). Maintain current strategies.")

st.markdown("---")
if st.button("Generate Word Report", type="primary"):
    st.success("Report generation ready (full implementation available on request)")

st.caption("Saul Damon High School • Professional Term Performance Dashboard")
