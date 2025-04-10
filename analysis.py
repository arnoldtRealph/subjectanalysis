import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches
import re
import traceback

# Streamlit Config
st.set_page_config(page_title="📊 Term Performance Dashboard", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #F4F6F9;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .custom-title {
            font-size: 30px;
            color: #003366;
        }
        h2, h3 {
            color: #003366;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("📊 Dashboard Options")
uploaded_file = st.sidebar.file_uploader("Upload TERM TEMPLATE Excel file", type=["xlsx"])
chart_type = st.sidebar.selectbox("Select Average Marks Chart Type", ["Bar", "Stacked Bar", "Pie"], index=0)

# Main content
st.header("SAUL DAMON HIGH SCHOOL")
st.subheader("Term Report Analysis")
st.info("Upload your TERM TEMPLATE.xlsx file in the sidebar to generate insights.")

# Process uploaded file
if uploaded_file:
    df = pd.read_excel(uploaded_file, header=None)
    
    # Identify grade sections
    grade_starts = df.index[df[0].str.contains("GRADE", na=False)].tolist()
    grades = ["Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_dfs = {}
    
    for i, grade in enumerate(grades):
        start_idx = grade_starts[i] + 2
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df)
        grade_df = df.iloc[start_idx:end_idx].dropna(how="all")
        grade_df.columns = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        
        # Clean the Subject column
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan")]
        
        # Convert numeric columns, treating NaN as 0 for levels
        grade_df[["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]] = grade_df[["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]].apply(pd.to_numeric, errors="coerce")
        grade_df[[f"LEVEL {i}" for i in range(1, 8)]] = grade_df[[f"LEVEL {i}" for i in range(1, 8)]].fillna(0)
        grade_dfs[grade] = grade_df

    # Analysis and Visualization
    for grade, gdf in grade_dfs.items():
        st.subheader(f"{grade} Analysis")
        
        st.write(f"{grade} Subject Performance")
        st.dataframe(gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style.format({"AVERAGE MARK": "{:.2f}", "TOTAL": "{:.0f}"}, na_rep="-"))

        st.write("Average Marks per Subject")
        fig, ax = plt.subplots(figsize=(12, 6))
        if chart_type == "Bar":
            sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="viridis", ax=ax, edgecolor="black")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("Subject", fontsize=12)
            ax.set_ylabel("Average Mark (%)", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        elif chart_type == "Stacked Bar":
            gdf.set_index("SUBJECT")[["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7"]].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("Subject", fontsize=12)
            ax.set_ylabel("Number of Students", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        elif chart_type == "Pie":
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(gdf)))
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(10)
            ax.axis('equal')
        ax.set_title(f"{grade} Average Marks", fontsize=14, pad=15)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Level Distribution per Subject")
        cols = st.columns(3)
        for idx, row in gdf.iterrows():
            subject = row["SUBJECT"]
            levels = row[["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7"]]
            total = row["TOTAL"]
            if not pd.isna(total) and total > 0 and levels.sum() > 0:
                with cols[idx % 3]:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    wedges, _ = ax.pie(
                        levels,
                        labels=None,
                        startangle=90,
                        colors=sns.color_palette("Set2", 7)
                    )
                    ax.legend(
                        labels=[f"Level {i} ({int(levels[i-1])})" for i in range(1, 8) if levels[i-1] > 0],
                        title="Levels",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1),
                        fontsize=8
                    )
                    ax.set_title(f"{subject}", fontsize=10, pad=10)
                    ax.axis('equal')
                    plt.tight_layout()
                    st.pyplot(fig)

        st.write("Pass/Fail Distribution per Subject")
        pass_counts = []
        fail_counts = []
        invalid_subjects = []
        subjects_for_pass_fail = []
        for _, row in gdf.iterrows():
            subject = row["SUBJECT"]
            total = row["TOTAL"]
            if pd.isna(total):
                invalid_subjects.append(subject)
                pass_counts.append(0)
                fail_counts.append(0)
                continue
            subjects_for_pass_fail.append(subject)
            if "Afrikaans HL" in subject or "Afrikaans FAL" in subject:
                fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                pass_count = total - fail_count
            elif subject == "Mathematics (Gr 09)" and grade == "Grade 9":
                fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                pass_count = total - fail_count
            else:
                fail_count = row["LEVEL 1"]
                pass_count = total - fail_count
            pass_counts.append(pass_count if not pd.isna(pass_count) else 0)
            fail_counts.append(fail_count if not pd.isna(fail_count) else 0)

        if invalid_subjects:
            st.warning(f"Warning: The following subjects in {grade} have missing total data and were excluded from pass/fail analysis: {', '.join(invalid_subjects)}")

        pass_fail_df = gdf.copy()
        pass_fail_df["PASSED"] = pass_counts
        pass_fail_df["FAILED"] = fail_counts

        if not pass_fail_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="bar", stacked=True, ax=ax, color=["#FF6B6B", "#4ECDC4"])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("Subject", fontsize=12)
            ax.set_ylabel("Number of Learners", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.set_title(f"{grade} Pass/Fail Distribution", fontsize=14, pad=15)
            
            for i, (passed, failed) in enumerate(zip(pass_fail_df["PASSED"], pass_fail_df["FAILED"])):
                total_height = passed + failed
                if pd.isna(total_height) or total_height == 0:
                    continue
                if failed > 0:
                    ax.text(i, failed / 2, f"{int(failed)}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                if passed > 0:
                    ax.text(i, failed + passed / 2, f"{int(passed)}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                ax.text(i, total_height + 0.5, f"{int(total_height)}", ha="center", va="bottom", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)

        st.write("Insights and Recommendations")
        for _, row in gdf.iterrows():
            subject = row["SUBJECT"]
            avg_mark = row["AVERAGE MARK"]
            total = row["TOTAL"]
            fail_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
            pass_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["PASSED"].values[0]
            fail_rate = (fail_count / total) * 100 if total > 0 else 0
            pass_rate = (pass_count / total) * 100 if total > 0 else 0
            
            st.markdown(f"- **{subject}**: Avg: {avg_mark:.2f}%, Pass Rate: {pass_rate:.2f}%, Fail Rate: {fail_rate:.2f}%")
            if fail_rate > 30:
                st.markdown(f"  - **Recommendation**: High fail rate ({fail_rate:.2f}%). Consider additional support or tutoring.")
            elif avg_mark < 50:
                st.markdown(f"  - **Recommendation**: Low average mark ({avg_mark:.2f}). Review teaching methods or resources.")
            else:
                st.markdown(f"  - **Recommendation**: Performing adequately (Pass Rate: {pass_rate:.2f}%). Maintain current strategies.")

    # Download Report as Word Document
    st.subheader("Download Report")
    def generate_word_report():
        try:
            doc = Document()
            doc.add_heading("Saul Damon High School Term Report", 0)
            doc.add_paragraph(f"Generated on {pd.Timestamp.now().strftime('%B %d, %Y')}")
            
            for grade, gdf in grade_dfs.items():
                doc.add_heading(f"{grade} Analysis", level=1)

                # Add Average Marks Chart
                doc.add_paragraph("Average Marks per Subject")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="viridis", ax=ax, edgecolor="black")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                ax.set_title(f"{grade} Average Marks", fontsize=14)
                ax.grid(True, axis="y", linestyle="--", alpha=0.7)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                doc.add_picture(buf, width=Inches(5.5))
                plt.close(fig)
                buf.close()

                # Add Pass/Fail Chart
                doc.add_paragraph("Pass/Fail Distribution per Subject")
                pass_counts = []
                fail_counts = []
                for _, row in gdf.iterrows():
                    subject = row["SUBJECT"]
                    total = row["TOTAL"]
                    if pd.isna(total):
                        pass_counts.append(0)
                        fail_counts.append(0)
                        continue
                    if "Afrikaans HL" in subject or "Afrikaans FAL" in subject:
                        fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                        pass_count = total - fail_count
                    elif subject == "Mathematics (Gr 09)" and grade == "Grade 9":
                        fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                        pass_count = total - fail_count
                    else:
                        fail_count = row["LEVEL 1"]
                        pass_count = total - fail_count
                    pass_counts.append(pass_count if not pd.isna(pass_count) else 0)
                    fail_counts.append(fail_count if not pd.isna(fail_count) else 0)

                pass_fail_df = gdf.copy()
                pass_fail_df["PASSED"] = pass_counts
                pass_fail_df["FAILED"] = fail_counts
                if not pass_fail_df.empty:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="bar", stacked=True, ax=ax, color=["#FF6B6B", "#4ECDC4"])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                    ax.set_title(f"{grade} Pass/Fail Distribution", fontsize=14)
                    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
                    for i, (passed, failed) in enumerate(zip(pass_fail_df["PASSED"], pass_fail_df["FAILED"])):
                        total_height = passed + failed
                        if pd.isna(total_height) or total_height == 0:
                            continue
                        if failed > 0:
                            ax.text(i, failed / 2, f"{int(failed)}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                        if passed > 0:
                            ax.text(i, failed + passed / 2, f"{int(passed)}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
                        ax.text(i, total_height + 0.5, f"{int(total_height)}", ha="center", va="bottom", fontsize=8)
                    buf = BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    doc.add_picture(buf, width=Inches(5.5))
                    plt.close(fig)
                    buf.close()

                # Add Level Distribution Pie Charts (sample of 3)
                doc.add_paragraph("Level Distribution (Sample)")
                for idx, row in gdf.head(3).iterrows():
                    subject = row["SUBJECT"]
                    levels = row[["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7"]]
                    if levels.sum() > 0:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.pie(levels, labels=None, startangle=90, colors=sns.color_palette("Set2", 7))
                        ax.legend(
                            labels=[f"Level {i} ({int(levels[i-1])})" for i in range(1, 8) if levels[i-1] > 0],
                            title="Levels",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1)
                        )
                        ax.set_title(f"{subject}", fontsize=10)
                        buf = BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        doc.add_picture(buf, width=Inches(2.5))
                        plt.close(fig)
                        buf.close()

                # Insights and Recommendations
                doc.add_paragraph("Insights and Recommendations")
                for _, row in gdf.iterrows():
                    subject = row["SUBJECT"]
                    avg_mark = row["AVERAGE MARK"]
                    total = row["TOTAL"]
                    fail_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
                    pass_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["PASSED"].values[0]
                    fail_rate = (fail_count / total) * 100 if total > 0 else 0
                    pass_rate = (pass_count / total) * 100 if total > 0 else 0
                    
                    doc.add_paragraph(f"{subject}: Avg: {avg_mark:.2f}%, Pass Rate: {pass_rate:.2f}%, Fail Rate: {fail_rate:.2f}%")
                    if fail_rate > 30:
                        doc.add_paragraph(f"  Recommendation: High fail rate ({fail_rate:.2f}%). Consider additional support or tutoring.")
                    elif avg_mark < 50:
                        doc.add_paragraph(f"  Recommendation: Low average mark ({avg_mark:.2f}). Review teaching methods or resources.")
                    else:
                        doc.add_paragraph(f"  Recommendation: Performing adequately (Pass Rate: {pass_rate:.2f}%). Maintain current strategies.")

            doc_stream = BytesIO()
            doc.save(doc_stream)
            doc_stream.seek(0)
            return doc_stream.getvalue()
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return None

    if st.button("Generate Word Report"):
        with st.spinner("Generating report..."):
            doc_data = generate_word_report()
            if doc_data:
                st.download_button(
                    label="Download Report",
                    data=doc_data,
                    file_name="term_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_button"
                )
                st.success("Report generated successfully! Click the button to download.")
            else:
                st.error("Failed to generate the report. Check the error message above.")