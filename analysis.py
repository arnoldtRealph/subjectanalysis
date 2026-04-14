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

# ====================== HIGH-READABILITY STYLING ======================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0A2540 0%, #112B4F 100%);
            color: #F8FAFF;
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 56px;
            font-weight: 700;
            background: linear-gradient(90deg, #00E6D8, #5EDFFF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }
        
        .sub-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 30px;
            color: #B0F5E8;
            text-align: center;
            margin-bottom: 50px;
            letter-spacing: 1px;
        }
        
        .section-header {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 28px;
            color: #00E6D8;
            border-bottom: 3px solid #00E6D8;
            padding-bottom: 14px;
            margin: 50px 0 30px 0;
        }
        
        .metric-card {
            background: rgba(0, 230, 216, 0.15);
            border: 1px solid #00E6D8;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #00E6D8, #5EDFFF);
            color: #0A2540;
            border-radius: 12px;
            padding: 14px 40px;
            font-weight: 600;
            font-size: 16.5px;
        }
        
        h1, h2, h3, h4, p, label, span, .stMarkdown {
            color: #F8FAFF !important;
        }
        
        .stDataFrame {
            font-size: 16px;
        }
        
        /* Welcome screen improvement */
        .welcome-text {
            font-size: 20px;
            color: #B0F5E8;
            text-align: center;
            max-width: 700px;
            margin: 0 auto;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("<h2 style='color:#00E6D8; text-align:center;'>TERM PERFORMANCE DASHBOARD</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE Excel File", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"], index=0)
    chart_type = st.selectbox("Average Marks Chart Type", ["Bar", "Stacked Bar", "Pie"], index=0)

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
        <div style='text-align:center; padding:80px 20px;'>
            <h2 style='color:#00E6D8;'>Welcome to the Term Performance Dashboard</h2>
            <p class='welcome-text'>
                Upload your <strong>TERM TEMPLATE.xlsx</strong> file from the sidebar to generate 
                detailed subject analysis, visualizations, and insights.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ====================== DATA PROCESSING ======================
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
        
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x).strip())
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan") & (grade_df["SUBJECT"] != "")]
        
        numeric_cols = ["AVERAGE MARK"] + [f"LEVEL {i}" for i in range(1, 8)] + ["TOTAL"]
        grade_df[numeric_cols] = grade_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        
        # Reliable TOTAL calculation
        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        grade_df["TOTAL"] = grade_df[level_cols].sum(axis=1)
        
        grade_dfs[grade] = grade_df
    
    return grade_dfs

grade_dfs = process_data(uploaded_file)

# ====================== MAIN DASHBOARD ======================
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
        st.metric("Total Learners", f"{int(gdf['TOTAL'].sum()):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Full-width Table
    st.subheader("Subject Performance Table")
    styled = gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style\
        .format({"AVERAGE MARK": "{:.1f}%", "TOTAL": "{:,.0f}"})\
        .background_gradient(cmap="Blues", subset=["AVERAGE MARK"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Full-width Main Chart
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

    # Level Distribution with Safety Check
    st.markdown('<p class="section-header">Level Distribution per Subject</p>', unsafe_allow_html=True)
    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    pie_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(gdf.iterrows()):
        subject = row["SUBJECT"]
        levels = row[level_cols]
        total_levels = levels.sum()
        
        with pie_cols[idx % 3]:
            if total_levels <= 0:
                st.info(f"**{subject}**\n\nNo level data available")
                continue
                
            fig, ax = plt.subplots(figsize=(5.8, 5.8))
            wedges, _ = ax.pie(levels, startangle=90, colors=sns.color_palette("Blues", 7))
            
            legend_labels = [f"Level {i} ({int(levels.iloc[i-1])})" 
                           for i in range(1, 8) if levels.iloc[i-1] > 0]
            
            ax.legend(wedges, legend_labels, title="Levels", 
                     loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=10)
            ax.set_title(subject, fontsize=12, color="#B0F5E8")
            ax.axis('equal')
            st.pyplot(fig)

    # Pass/Fail
    st.markdown('<p class="section-header">Pass / Fail Distribution</p>', unsafe_allow_html=True)
    
    pass_counts, fail_counts = [], []
    for _, row in gdf.iterrows():
        total = row["TOTAL"]
        if total <= 0:
            pass_counts.append(0)
            fail_counts.append(0)
            continue
            
        if any(x in row["SUBJECT"] for x in ["Afrikaans HL", "Afrikaans FAL", "Afrikaans HT"]) or \
           ("Mathematics (Gr 09)" in row["SUBJECT"] and grade == "Grade 9"):
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
        if total == 0:
            continue
        failed = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
        fail_rate = (failed / total * 100)
        
        if fail_rate > 30:
            st.error(f"🔴 **{subject}** — High fail rate ({fail_rate:.1f}%). Targeted intervention recommended.")
        elif avg < 50:
            st.warning(f"🟠 **{subject}** — Low average ({avg:.1f}%). Review teaching methods.")
        else:
            st.success(f"🟢 **{subject}** — Strong performance (Pass rate: {100-fail_rate:.1f}%). Continue current strategies.")

st.markdown("---")
st.caption("Saul Damon High School • Professional Term Performance Dashboard")
