"""Creates 8 more SYNTHETIC resumes (fictional people) for testing and demo."""
import os
from docx import Document

OUTPUT_DIR = os.path.join("data", "sample_resumes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

resumes = {
    "Sneha_Iyer.docx": {
        "header": ["Sneha Iyer", "sneha.iyer@example.com | +91 90123 45678"],
        "sections": {
            "Summary": "Machine learning engineer building NLP and text classification solutions.",
            "Skills": "Python, Scikit-learn, TensorFlow, Keras, NLP, spaCy, NLTK, Pandas, NumPy, SQL, Docker, Git",
            "Experience": "Built a text classification system using TF-IDF and logistic regression. "
                          "Cleaned and analysed data, built predictive models and presented insights to stakeholders.",
            "Education": "B.Tech Computer Science, 2025",
        },
    },
    "Karan_Mehta.docx": {   # keyword-heavy resume, almost no description
        "header": ["Karan Mehta", "karan.mehta@example.com | 9988776655"],
        "sections": {
            "Summary": "Fresher looking for opportunities.",
            "Skills": "Python, Pandas, NumPy, Scikit-learn, SQL, NLP, Machine Learning, TF-IDF, Git, "
                      "Streamlit, Matplotlib, Docker, AWS, Java, Tableau",
            "Experience": "Internship.",
            "Education": "B.Tech, 2026",
        },
    },
    "Divya_Nair.docx": {    # describes the work in prose, lists few skills
        "header": ["Divya Nair", "divya.nair@example.com | +91 91234 00011"],
        "sections": {
            "Summary": "Data scientist fresher looking for a data scientist role.",
            "Skills": "Python",
            "Experience": "Built solutions that clean and analyse data, build predictive models, "
                          "create dashboards and present insights to stakeholders.",
            "Education": "B.Tech in Computer Science or a related field, 2026",
        },
    },
    "Vikram_Singh.docx": {
        "header": ["Vikram Singh", "vikram.singh@example.com | +91 98100 22334"],
        "sections": {
            "Summary": "Data engineer working on large scale data pipelines.",
            "Skills": "Python, SQL, PostgreSQL, ETL, Spark, Hadoop, AWS, Docker, Git, Linux",
            "Experience": "Designed ETL pipelines on AWS. Optimised SQL queries and managed PostgreSQL databases.",
            "Education": "B.Tech Computer Science, 2023",
        },
    },
    "Neha_Gupta.docx": {
        "header": ["Neha Gupta", "neha.gupta@example.com | 9811122233"],
        "sections": {
            "Summary": "Frontend developer creating responsive user interfaces.",
            "Skills": "HTML, CSS, JavaScript, React, Git",
            "Experience": "Built responsive websites and single page applications using React.",
            "Education": "BCA, 2024",
        },
    },
    "Arjun_Das.docx": {
        "header": ["Arjun Das", "arjun.das@example.com | +91 97000 11122"],
        "sections": {
            "Summary": "Aspiring data analyst with academic projects in data analysis.",
            "Skills": "Python, Pandas, Excel, Matplotlib, SQL, Statistics, Git",
            "Experience": "Analysed student performance data using Pandas and created charts using Matplotlib.",
            "Education": "B.Sc Mathematics, 2026",
        },
    },
    "Pooja_Reddy.docx": {
        "header": ["Pooja Reddy", "pooja.reddy@example.com | 9000011122"],
        "sections": {
            "Summary": "Business analyst experienced in reporting and dashboards.",
            "Skills": "Excel, Power BI, SQL, Tableau, Data Analysis",
            "Experience": "Created weekly sales dashboards in Power BI and Tableau for management.",
            "Education": "MBA, 2022",
        },
    },
    "Rohit_Patil.docx": {
        "header": ["Rohit Patil", "rohit.patil@example.com | +91 98220 33445"],
        "sections": {
            "Summary": "Backend developer building enterprise Java applications.",
            "Skills": "Java, Spring Boot, MySQL, REST API, Docker, Git",
            "Experience": "Developed REST APIs using Spring Boot and MySQL for a banking application.",
            "Education": "B.Tech Information Technology, 2023",
        },
    },
}

for filename, content in resumes.items():
    doc = Document()
    for line in content["header"]:
        doc.add_paragraph(line)
    for heading, body in content["sections"].items():
        doc.add_heading(heading, level=2)
        doc.add_paragraph(body)
    doc.save(os.path.join(OUTPUT_DIR, filename))
    print(f"Created: {filename}")