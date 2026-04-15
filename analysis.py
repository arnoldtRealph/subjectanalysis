import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Saul Damon High School | Term Analysis",
    layout="wide",
    page_icon="📊"
)

# ====================== CLEAN LIGHT STYLING ======================
st.markdown("""
    <style>
        .stApp {
            background-color: #F5F7FA;
            color: #1F2937;
            font-family: Arial, sans-serif;
        }
        
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            color: #1F2937;
        }
        
        .sub-title {
            font-size: 22px;
            text-align: center;
            color: #4B5563;
            margin-bottom: 30px;
        }
        
        .section-header {
            font-size: 22px;
            font-weight: 600;
            color: #2563EB;
            border-bottom: 2px solid #2563EB;
            padding-bottom: 8px;
            margin: 35px 0 20px 0;
        }
        
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 18px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Dashboard Settings")
    uploaded_file = st.file_uploader("Upload TERM TEMPLATE Excel File", type=["xlsx"])
    term = st.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"])
    chart_type = st.selectbox("Chart Type", ["Bar", "Stacked Bar", "Pie"])

# ====================== HEADER ======================
st.markdown('<p class="main-title">SAUL DAMON HIGH SCHOOL</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{term} Performance Analysis</p>', unsafe_allow_html=True)

if not uploaded_file:
    st.info("Upload your TERM TEMPLATE Excel file to begin.")
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
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"] != ""]
        
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
        
    st.markdown(f'<p class="section-header">{grade}</p>', unsafe_allow_html=True)
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", len(gdf))
    c2.metric("Avg Mark", f"{gdf['AVERAGE MARK'].mean():.1f}%")
    c3.metric("Learners", int(gdf["TOTAL"].sum()))

    # Table
    st.dataframe(
        gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]]
        .sort_values("AVERAGE MARK", ascending=False),
        use_container_width=True
    )

    # ====================== CHART ======================
    st.subheader("Average Marks")

    if chart_type == "Pie":
        fig, ax = plt.subplots(figsize=(7, 7))

        ax.pie(
            gdf["AVERAGE MARK"],
            labels=gdf["SUBJECT"],
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel")
        )
        ax.axis('equal')

    else:
        fig, ax = plt.subplots(figsize=(12, 6))

        if chart_type == "Bar":
            sns.barplot(
                x="AVERAGE MARK",
                y="SUBJECT",
                data=gdf,
                color="#3B82F6",
                ax=ax
            )
        else:
            level_cols = [f"LEVEL {i}" for i in range(1, 8)]
            gdf.set_index("SUBJECT")[level_cols].plot(
                kind="barh",
                stacked=True,
                colormap="Blues",
                ax=ax
            )

    plt.tight_layout()
    st.pyplot(fig)

    # ====================== LEVEL DISTRIBUTION ======================
    st.markdown('<p class="section-header">Level Distribution</p>', unsafe_allow_html=True)

    level_cols = [f"LEVEL {i}" for i in range(1, 8)]
    cols = st.columns(3)

    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 3]:
            levels = row[level_cols]
            if levels.sum() == 0:
                st.write(f"{row['SUBJECT']}: No data")
                continue

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                levels,
                autopct='%1.0f%%',
                startangle=90,
                colors=sns.color_palette("Blues")
            )
            ax.set_title(row["SUBJECT"], fontsize=10)
            ax.axis('equal')
            st.pyplot(fig)

    # ====================== INSIGHTS ======================
    st.markdown('<p class="section-header">Insights</p>', unsafe_allow_html=True)

    for _, row in gdf.iterrows():
        if row["TOTAL"] == 0:
            continue

        fail = row["LEVEL 1"]
        fail_rate = fail / row["TOTAL"] * 100

        if fail_rate > 30:
            st.error(f"{row['SUBJECT']}: High fail rate ({fail_rate:.1f}%)")
        elif row["AVERAGE MARK"] < 50:
            st.warning(f"{row['SUBJECT']}: Low average ({row['AVERAGE MARK']:.1f}%)")
        else:
            st.success(f"{row['SUBJECT']}: Performing well")

st.caption("Saul Damon High School • Clean Performance Dashboard")
