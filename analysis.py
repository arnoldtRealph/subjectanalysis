import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
import traceback

# Streamlit Config
st.set_page_config(page_title="📊 Term Performance Dashboard", layout="wide")

# Custom Styling with Professional Colors
st.markdown("""
    <style>
        .stApp {
            background-color: #F5F6F5;
            color: #2E2E2E;
            font-family: 'Helvetica', sans-serif;
        }
        .custom-title {
            font-size: 36px;
            color: #1A3C5E;
            font-weight: bold;
        }
        .custom-subheader {
            font-size: 24px;
            color: #1A3C5E;
        }
        h2, h3 {
            color: #1A3C5E;
        }
        .stButton>button {
            background-color: #1A3C5E;
            color: #FFFFFF;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #2A567D;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("📊 Dashboard Options")
st.sidebar.markdown("<p style='font-size: 18px;'>Configure your report below:</p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload TERM TEMPLATE Excel file", type=["xlsx"])
term = st.sidebar.selectbox("Select Term", ["Term 1", "Term 2", "Term 3", "Term 4"], index=0)
chart_type = st.sidebar.selectbox("Select Average Marks Chart Type", ["Bar", "Stacked Bar", "Pie"], index=0)

# Main content with larger text
st.markdown('<p class="custom-title">Saul Damon High School</p>', unsafe_allow_html=True)
st.markdown(f'<p class="custom-subheader">{term} Report Analysis</p>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px;'>Upload your TERM TEMPLATE.xlsx file in the sidebar to generate insights.</p>", unsafe_allow_html=True)

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
        st.markdown(f"<h2>{grade} Analysis</h2>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-size: 18px;'>Subject Performance</p>", unsafe_allow_html=True)
        st.dataframe(gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style.format({"AVERAGE MARK": "{:.2f}", "TOTAL": "{:.0f}"}, na_rep="-"))

        st.markdown("<p style='font-size: 18px;'>Average Marks per Subject</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        if chart_type == "Bar":
            sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues", ax=ax, edgecolor="black")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("Subject", fontsize=12)
            ax.set_ylabel("Average Mark (%)", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        elif chart_type == "Stacked Bar":
            gdf.set_index("SUBJECT")[["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7"]].plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("Subject", fontsize=12)
            ax.set_ylabel("Number of Students", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        elif chart_type == "Pie":
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Blues", len(gdf)))
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(10)
            ax.axis('equal')
        ax.set_title(f"{grade} Average Marks", fontsize=14, pad=15)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("<p style='font-size: 18px;'>Level Distribution per Subject</p>", unsafe_allow_html=True)
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
                        colors=sns.color_palette("Purples", 7)
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

        st.markdown("<p style='font-size: 18px;'>Pass/Fail Distribution per Subject</p>", unsafe_allow_html=True)
        pass_counts = []
        fail_counts = []
        invalid_subjects = []
        for _, row in gdf.iterrows():
            subject = row["SUBJECT"]
            total = row["TOTAL"]
            if pd.isna(total):
                invalid_subjects.append(subject)
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

        if invalid_subjects:
            st.warning(f"Warning: The following subjects in {grade} have missing total data and were excluded from pass/fail analysis: {', '.join(invalid_subjects)}")

        pass_fail_df = gdf.copy()
        pass_fail_df["PASSED"] = pass_counts
        pass_fail_df["FAILED"] = fail_counts

        if not pass_fail_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="bar", stacked=True, ax=ax, color=["#D9534F", "#5BC0DE"])
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

        st.markdown("<p style='font-size: 18px;'>Insights and Recommendations</p>", unsafe_allow_html=True)
        for _, row in gdf.iterrows():
            subject = row["SUBJECT"]
            avg_mark = row["AVERAGE MARK"]
            total = row["TOTAL"]
            fail_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
            pass_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["PASSED"].values[0]
            fail_rate = (fail_count / total) * 100 if total > 0 else 0
            pass_rate = (pass_count / total) * 100 if total > 0 else 0
            
            st.markdown(f"<p style='font-size: 16px;'><b>{subject}</b>: Avg: {avg_mark:.2f}%, Pass Rate: {pass_rate:.2f}%, Fail Rate: {fail_rate:.2f}%</p>", unsafe_allow_html=True)
            if fail_rate > 30:
                st.markdown(f"<p style='font-size: 14px;'>  - <b>Recommendation</b>: High fail rate ({fail_rate:.2f}%). Consider additional support or tutoring.</p>", unsafe_allow_html=True)
            elif avg_mark < 50:
                st.markdown(f"<p style='font-size: 14px;'>  - <b>Recommendation</b>: Low average mark ({avg_mark:.2f}). Review teaching methods or resources.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='font-size: 14px;'>  - <b>Recommendation</b>: Performing adequately (Pass Rate: {pass_rate:.2f}%). Maintain current strategies.</p>", unsafe_allow_html=True)

    # Download Report as Word Document
    st.markdown("<h2>Download Report</h2>", unsafe_allow_html=True)
    def generate_word_report():
        try:
            doc = Document()
            
            # Title Section
            title = doc.add_heading(f"Saul Damon High School {term} Report", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            title.paragraph_format.space_after = Pt(12)
            
            date_paragraph = doc.add_paragraph(f"Generated on {pd.Timestamp.now().strftime('%B %d, %Y')}")
            date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_paragraph.paragraph_format.space_after = Pt(24)
            
            for grade, gdf in grade_dfs.items():
                # Grade Section Heading
                grade_heading = doc.add_heading(f"{grade} Analysis", level=1)
                grade_heading.paragraph_format.space_before = Pt(18)
                grade_heading.paragraph_format.space_after = Pt(12)

                # Average Marks Chart
                doc.add_paragraph("Average Marks per Subject", style='Heading 2')
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x="SUBJECT", y="AVERAGE MARK", data=gdf, palette="Blues", ax=ax, edgecolor="black")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                ax.set_title(f"{grade} Average Marks", fontsize=12)
                ax.grid(True, axis="y", linestyle="--", alpha=0.7)
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                doc.add_picture(buf, width=Inches(6))
                plt.close(fig)
                buf.close()
                doc.add_paragraph().add_run().add_break()

                # Pass/Fail Chart
                doc.add_paragraph("Pass/Fail Distribution per Subject", style='Heading 2')
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
                    fig, ax = plt.subplots(figsize=(10, 5))
                    pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="bar", stacked=True, ax=ax, color=["#D9534F", "#5BC0DE"])
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                    ax.set_title(f"{grade} Pass/Fail Distribution", fontsize=12)
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
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    buf.seek(0)
                    doc.add_picture(buf, width=Inches(6))
                    plt.close(fig)
                    buf.close()
                    doc.add_paragraph().add_run().add_break()

                # Level Distribution Pie Charts (Sample of up to 3)
                doc.add_paragraph("Level Distribution (Sample)", style='Heading 2')
                num_subjects = min(3, len(gdf))  # Limit to available subjects, max 3
                if num_subjects > 0:
                    pie_table = doc.add_table(rows=1, cols=num_subjects)
                    pie_table.autofit = True
                    for idx, (index, row) in enumerate(gdf.iterrows()):  # Use enumerate with iterrows
                        if idx >= num_subjects:  # Break if we've filled the table
                            break
                        subject = row["SUBJECT"]
                        levels = row[["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7"]]
                        if levels.sum() > 0:
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.pie(levels, labels=None, startangle=90, colors=sns.color_palette("Purples", 7))
                            ax.legend(
                                labels=[f"Level {i} ({int(levels[i-1])})" for i in range(1, 8) if levels[i-1] > 0],
                                title="Levels",
                                loc="center left",
                                bbox_to_anchor=(1, 0, 0.5, 1),
                                fontsize=8
                            )
                            ax.set_title(f"{subject}", fontsize=10)
                            buf = BytesIO()
                            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                            buf.seek(0)
                            cell = pie_table.rows[0].cells[idx]
                            cell.paragraphs[0].add_run().add_picture(buf, width=Inches(2))
                            plt.close(fig)
                            buf.close()

                # Insights and Recommendations
                doc.add_paragraph("Insights and Recommendations", style='Heading 2')
                for _, row in gdf.iterrows():
                    subject = row["SUBJECT"]
                    avg_mark = row["AVERAGE MARK"]
                    total = row["TOTAL"]
                    fail_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["FAILED"].values[0]
                    pass_count = pass_fail_df[pass_fail_df["SUBJECT"] == subject]["PASSED"].values[0]
                    fail_rate = (fail_count / total) * 100 if total > 0 else 0
                    pass_rate = (pass_count / total) * 100 if total > 0 else 0
                    
                    p = doc.add_paragraph()
                    p.add_run(f"{subject}: ").bold = True
                    p.add_run(f"Avg: {avg_mark:.2f}%, Pass Rate: {pass_rate:.2f}%, Fail Rate: {fail_rate:.2f}%")
                    if fail_rate > 30:
                        doc.add_paragraph(f"  Recommendation: High fail rate ({fail_rate:.2f}%). Consider additional support or tutoring.", style='List Bullet')
                    elif avg_mark < 50:
                        doc.add_paragraph(f"  Recommendation: Low average mark ({avg_mark:.2f}). Review teaching methods or resources.", style='List Bullet')
                    else:
                        doc.add_paragraph(f"  Recommendation: Performing adequately (Pass Rate: {pass_rate:.2f}%). Maintain current strategies.", style='List Bullet')

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
                    file_name=f"{term.lower().replace(' ', '_')}_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_button"
                )
                st.success("Report generated successfully! Click the button to download.")
            else:
                st.error("Failed to generate the report. Check the error message above.")