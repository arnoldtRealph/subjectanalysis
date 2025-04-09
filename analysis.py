import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from docx import Document
from docx.shared import Inches
import re

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
        start_idx = grade_starts[i] + 5  # Adjust for headers
        end_idx = grade_starts[i + 1] if i + 1 < len(grade_starts) else len(df)
        grade_df = df.iloc[start_idx:end_idx].dropna(how="all")
        grade_df.columns = ["SUBJECT", "AVERAGE MARK", "LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7", "TOTAL"]
        
        # Clean the.Subject column
        grade_df["SUBJECT"] = grade_df["SUBJECT"].astype(str).str.strip()
        # Remove non-printable characters
        grade_df["SUBJECT"] = grade_df["SUBJECT"].apply(lambda x: re.sub(r'[^\x20-\x7E]', '', x))
        grade_df = grade_df[grade_df["SUBJECT"].notna() & (grade_df["SUBJECT"] != "nan")]
        
        # Convert numeric columns, treating NaN as 0 for levels
        grade_df[["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]] = grade_df[["AVERAGE MARK", "TOTAL"] + [f"LEVEL {i}" for i in range(1, 8)]].apply(pd.to_numeric, errors="coerce")
        grade_df[[f"LEVEL {i}" for i in range(1, 8)]] = grade_df[[f"LEVEL {i}" for i in range(1, 8)]].fillna(0)
        grade_dfs[grade] = grade_df

    # Analysis and Visualization
    for grade, gdf in grade_dfs.items():
        st.subheader(f"{grade} Analysis")
        
        # Summary Table
        st.write(f"{grade} Subject Performance")
        st.dataframe(gdf[["SUBJECT", "AVERAGE MARK", "TOTAL"]].style.format({"AVERAGE MARK": "{:.2f}", "TOTAL": "{:.0f}"}, na_rep="-"))

        # Chart for Average Marks
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
            avg_total = gdf["AVERAGE MARK"].sum()
            wedges, texts, autotexts = ax.pie(gdf["AVERAGE MARK"], labels=gdf["SUBJECT"], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(gdf)))
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(10)
            ax.axis('equal')
        ax.set_title(f"{grade} Average Marks", fontsize=14, pad=15)
        plt.tight_layout()
        st.pyplot(fig)

        # Pie Charts for Level Distribution per Subject with Key
        st.write("Level Distribution per Subject")
        cols = st.columns(3)  # Display 3 pie charts per row
        for idx, row in gdf.iterrows():
            subject = row["SUBJECT"]
            levels = row[["LEVEL 1", "LEVEL 2", "LEVEL 3", "LEVEL 4", "LEVEL 5", "LEVEL 6", "LEVEL 7"]]
            total = row["TOTAL"]
            if not pd.isna(total) and total > 0 and levels.sum() > 0:  # Only plot if there are students
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

        # Pass/Fail Chart with Counts (excluding Afrikaans subjects)
        st.write("Pass/Fail Distribution per Subject (Excluding Afrikaans)")
        pass_counts = []
        fail_counts = []
        invalid_subjects = []
        subjects_for_pass_fail = []
        for _, row in gdf.iterrows():
            subject = row["SUBJECT"]
            total = row["TOTAL"]
            # Skip if total is NaN
            if pd.isna(total):
                invalid_subjects.append(subject)
                pass_counts.append(0)
                fail_counts.append(0)
                continue
            # Skip Afrikaans subjects
            if "Afrikaans" in subject:
                pass_counts.append(0)
                fail_counts.append(0)
                continue
            subjects_for_pass_fail.append(subject)
            # Calculate fail count based on grade and subject
            if grade == "Grade 9":
                if subject == "English Home Language (Gr 09)":
                    fail_count = row[["LEVEL 1", "LEVEL 2", "LEVEL 3"]].sum()
                elif subject == "Mathematics (Gr 09)":
                    fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                else:
                    fail_count = row["LEVEL 1"]
            elif grade == "Grade 10":
                if subject == "English HL (Gr 10)":
                    fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                else:
                    fail_count = row["LEVEL 1"]
            elif grade == "Grade 11":
                if subject == "English HL (Gr 11)":
                    fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                else:
                    fail_count = row["LEVEL 1"]
            elif grade == "Grade 12":
                if subject == "English HL (Gr 12)":
                    fail_count = row[["LEVEL 1", "LEVEL 2"]].sum()
                else:
                    fail_count = row["LEVEL 1"]
            pass_count = total - fail_count
            pass_counts.append(pass_count if not pd.isna(pass_count) else 0)
            fail_counts.append(fail_count if not pd.isna(fail_count) else 0)

        if invalid_subjects:
            st.warning(f"Warning: The following subjects in {grade} have missing total data and were excluded from pass/fail analysis: {', '.join(invalid_subjects)}")

        # Create a new dataframe for pass/fail chart, excluding Afrikaans subjects
        pass_fail_df = gdf[gdf["SUBJECT"].isin(subjects_for_pass_fail)].copy()
        pass_fail_df["PASSED"] = [pass_counts[i] for i in range(len(gdf)) if gdf.iloc[i]["SUBJECT"] in subjects_for_pass_fail]
        pass_fail_df["FAILED"] = [fail_counts[i] for i in range(len(gdf)) if gdf.iloc[i]["SUBJECT"] in subjects_for_pass_fail]

        if not pass_fail_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="bar", stacked=True, ax=ax, color=["#FF6B6B", "#4ECDC4"])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            ax.set_xlabel("Subject", fontsize=12)
            ax.set_ylabel("Number of Learners", fontsize=12)
            ax.grid(True, axis="y", linestyle="--", alpha=0.7)
            ax.set_title(f"{grade} Pass/Fail Distribution (Excluding Afrikaans)", fontsize=14, pad=15)
            
            # Add text annotations
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
        else:
            st.write("No subjects available for pass/fail analysis after excluding Afrikaans.")

        # Ranking and Recommendations (including Afrikaans)
        sorted_df = gdf[["SUBJECT", "AVERAGE MARK"]].sort_values("AVERAGE MARK", ascending=False).reset_index(drop=True)
        sorted_df.index = sorted_df.index + 1  # Start numbering from 1
        st.write(f"{grade} Subjects Ranked (Best to Worst)")
        st.dataframe(sorted_df.style.format({"AVERAGE MARK": "{:.2f}"}, na_rep="-"))

        best_subject = sorted_df.iloc[0]["SUBJECT"] if not sorted_df.empty else "N/A"
        worst_subject = sorted_df.iloc[-1]["SUBJECT"] if not sorted_df.empty else "N/A"
        avg_mark = gdf["AVERAGE MARK"].mean()
        
        st.write("Insights and Recommendations")
        st.markdown(f"- **Best Performing Subject**: {best_subject} (Avg: {sorted_df.iloc[0]['AVERAGE MARK']:.2f}%)" if not sorted_df.empty else "- **Best Performing Subject**: N/A")
        st.markdown(f"- **Worst Performing Subject**: {worst_subject} (Avg: {sorted_df.iloc[-1]['AVERAGE MARK']:.2f}%)" if not sorted_df.empty else "- **Worst Performing Subject**: N/A")
        st.markdown(f"- **Grade Average**: {avg_mark:.2f}%" if not pd.isna(avg_mark) else "- **Grade Average**: N/A")
        
        if pd.isna(avg_mark) or avg_mark < 50:
            st.markdown(f"- **Recommendation**: Intensive support needed for {worst_subject} (e.g., extra classes, tutoring).")
        elif avg_mark < 70:
            st.markdown(f"- **Recommendation**: Targeted improvement for {worst_subject} while maintaining strengths.")
        else:
            st.markdown(f"- **Recommendation**: Excellent performance! Promote advanced learning in {best_subject}.")

    # Download Report
    st.subheader("Download Report")
    if st.button("Generate Word Report"):
        doc = Document()
        doc.add_heading("Saul Damon High School Term Report", 0)
        doc.add_paragraph(f"Generated on {pd.Timestamp.now().strftime('%B %d, %Y')}")
        
        for grade, gdf in grade_dfs.items():
            doc.add_heading(f"{grade} Analysis", level=1)
            doc.add_paragraph("Subject Performance Summary")
            table = doc.add_table(rows=len(gdf) + 1, cols=3)
            table.style = "Table Grid"
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = "Subject"
            hdr_cells[1].text = "Average Mark"
            hdr_cells[2].text = "Total Students"
            for i, row in enumerate(gdf.itertuples(), 1):
                row_cells = table.rows[i].cells
                row_cells[0].text = row.SUBJECT
                row_cells[1].text = f"{row.AVERAGE_MARK:.2f}" if not pd.isna(row.AVERAGE_MARK) else "-"
                row_cells[2].text = str(int(row.TOTAL)) if not pd.isna(row.TOTAL) else "-"

            # Add Average Marks Chart
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

            # Add Pass/Fail Chart (excluding Afrikaans)
            pass_fail_df = gdf[gdf["SUBJECT"].isin(subjects_for_pass_fail)].copy()
            pass_fail_df["PASSED"] = [pass_counts[i] for i in range(len(gdf)) if gdf.iloc[i]["SUBJECT"] in subjects_for_pass_fail]
            pass_fail_df["FAILED"] = [fail_counts[i] for i in range(len(gdf)) if gdf.iloc[i]["SUBJECT"] in subjects_for_pass_fail]
            if not pass_fail_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                pass_fail_df.set_index("SUBJECT")[["FAILED", "PASSED"]].plot(kind="bar", stacked=True, ax=ax, color=["#FF6B6B", "#4ECDC4"])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
                ax.set_title(f"{grade} Pass/Fail Distribution (Excluding Afrikaans)", fontsize=14)
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

            # Add Level Distribution Pie Charts (sample of 3, including Afrikaans)
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

            # Recommendations (including Afrikaans)
            sorted_df = gdf[["SUBJECT", "AVERAGE MARK"]].sort_values("AVERAGE MARK", ascending=False).reset_index(drop=True)
            sorted_df.index = sorted_df.index + 1
            doc.add_paragraph(f"Best: {sorted_df.iloc[0]['SUBJECT']} ({sorted_df.iloc[0]['AVERAGE MARK']:.2f}%)" if not sorted_df.empty else "Best: N/A")
            doc.add_paragraph(f"Worst: {sorted_df.iloc[-1]['SUBJECT']} ({sorted_df.iloc[-1]['AVERAGE MARK']:.2f}%)" if not sorted_df.empty else "Worst: N/A")
            doc.add_paragraph(f"Grade Avg: {gdf['AVERAGE MARK'].mean():.2f}%" if not pd.isna(gdf["AVERAGE MARK"].mean()) else "Grade Avg: N/A")

        doc_stream = BytesIO()
        doc.save(doc_stream)
        doc_stream.seek(0)
        st.download_button(
            label="Download Report",
            data=doc_stream,
            file_name="term_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )