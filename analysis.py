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

# ====================== CLEAN PROFESSIONAL STYLING ======================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .stApp {
            background-color: #0F1C2E;
            color: #E8F1F8;
            font-family: 'Inter', sans-serif;
        }
        
        .main-title {
            font-size: 48px;
            font-weight: 700;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        
        .sub-title {
            font-size: 26px;
            color: #A3C4D4;
            text-align: center;
            margin-bottom: 40px;
        }
        
        .section-header {
            font-size: 24px;
            font-weight: 600;
            color: #4ECDC4;
            border-bottom: 2px solid #4ECDC4;
            padding-bottom: 10px;
            margin: 45px 0 25px 0;
        }
        
        .metric-card {
            background-color: #1A2A44;
            border: 1px solid #4ECDC4;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .stButton>button {
            background-color: #4ECDC4;
            color: #0F1C2E;
            border: none;
            border-radius: 8px;
            padding: 12px 32px;
            font-weight: 600;
            font-size: 16px;
        }
        
        .stButton>button:hover {
            background-color: #3EB8A8;
        }
        
        h1, h2, h3, h4, p, label, span {
            color: #E8F1F8;
        }
        
        .stDataFrame {
            font-size: 15.5px;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("<h2 style='color:#4ECDC4; text-align:center;'>TERM PERFORMANCE DASHBOARD</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE Excel File", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"], index=0)
    chart_type = st.selectbox("Average Marks Chart Type", ["Bar", "Stacked Bar", "Pie"], index=0)

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.markdown("""
        <div style='text-align:center; padding:90px 20px;'>
            <h2 style='color:#FFFFFF;'>Welcome</h2>
            <p style='font-size:19px; color:#A3C4D4; max-width:680px; margin:0 auto;'>
                Upload your <strong>TERM TEMPLATE.xlsx</strong> file from the sidebar to generate 
                subject-level performance analysis and insights.
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
        
        level_cols = [f"LEVEL {i}" for i in range(1, 8)]
        grade_df["TOTAL"] = grade_df[level_cols].sum(axis=1)
        
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
        st.metric("Number of Subjects", len(gdf))
        st.markdown('</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_mark = gdf["AVERAGE MARK"].mean()
        st.metric("Overall Average Mark", f"{avg_mark:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Learners", f"{int(gdf['TOTAL'].sum()):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Table
    st.subheader("Subject Performance Table")
    styled = gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style\
        .format({"AVERAGE MARK": "{:.1f}%", "TOTAL": "{:,.0f}"})\
        .background_gradient(cmap="Blues", subset=["AVERAGE MARK"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ====================== CHART ======================
    st.subheader("Average Marks Visualization")
    
    if chart_type == "Pie":
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_alpha(0)
        ax.set_facecolor("#16243A")

        colors = sns.color_palette("blend:#4ECDC4,#1A2A44", len(gdf))

        wedges, texts, autotexts = ax.pie(
            gdf["AVERAGE MARK"],
            labels=gdf["SUBJECT"],
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.8,
            colors=colors,
            wedgeprops=dict(width=0.4, edgecolor='white')
        )

        for text in texts:
            text.set_fontsize(10)

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(10)
            autotext.set_weight("bold")

        ax.text(0, 0, f"{grade}\nAvg Marks", ha='center', va='center',
                fontsize=14, weight='bold', color="#E8F1F8")

        ax.axis('equal')
        ax.set_title(f"{grade} — Average Marks per Subject", fontsize=15, pad=20)
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(14, 7.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("#16243A")

        if chart_type == "Bar":
            sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues_d", ax=ax)
        else:
            level_cols = [f"LEVEL {i}" for i in range(1, 8)]
            gdf.set_index("SUBJECT")[level_cols].plot(kind="bar", stacked=True, ax=ax, colormap="Blues")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title(f"{grade} — Average Marks per Subject", fontsize=15, pad=20)
        plt.tight_layout()
        st.pyplot(fig)

    # ====================== LEVEL DONUTS ======================
    st.markdown('<p class="section-header">Level Distribution per Subject</p>', unsafe_allow_html=True)
    
    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    pie_cols = st.columns(3)
    
    for idx, (_, row) in enumerate(gdf.iterrows()):
        subject = row["SUBJECT"]
        levels = row[level_cols]
        total_levels = levels.sum()
        
        with pie_cols[idx % 3]:
            if total_levels <= 0:
                st.info(f"**{subject}**\n\nNo level data")
                continue
                
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_alpha(0)
            ax.set_facecolor("#16243A")

            colors = sns.color_palette("coolwarm", 7)

            wedges, texts, autotexts = ax.pie(
                levels,
                autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
                startangle=90,
                pctdistance=0.75,
                colors=colors,
                wedgeprops=dict(width=0.45, edgecolor='white')
            )

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontsize(9)
                autotext.set_weight("bold")

            ax.text(0, 0, f"{int(total_levels)}\nLearners",
                    ha='center', va='center',
                    fontsize=11, weight='bold', color="#E8F1F8")

            ax.set_title(subject, fontsize=12, pad=12)
            ax.axis('equal')
            st.pyplot(fig)

    # ====================== PASS FAIL ======================
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
    ax.set_title(f"{grade} — Pass/Fail Distribution", fontsize=15, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # ====================== INSIGHTS ======================
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
            st.error(f"**{subject}** — High fail rate ({fail_rate:.1f}%). Targeted support is recommended.")
        elif avg < 50:
            st.warning(f"**{subject}** — Low average mark ({avg:.1f}%). Review teaching methods.")
        else:
            st.success(f"**{subject}** — Good performance. Maintain current strategies.")

st.markdown("---")
st.caption("Saul Damon High School • Term Performance Analysis Dashboard")
