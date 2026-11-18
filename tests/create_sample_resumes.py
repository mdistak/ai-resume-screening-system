"""Creates 3 fake sample DOCX resumes in data/sample_resumes/ for testing."""
import os
from docx import Document

OUTPUT_DIR = os.path.join("data", "sample_resumes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

resumes = {
    "Rahul_Sharma.docx": {
        "header": ["Rahul Sharma", "Email: rahul.sharma@example.com | Phone: +91 98765 43210"],
        "sections": {
            "Summary": "Data Science graduate with hands-on experience in machine learning and NLP.",
            "Skills": "Python, Pandas, NumPy, Scikit-learn, SQL, NLP, TF-IDF, Streamlit, Git, Matplotlib",
            "Experience": "6-month Data Science intern. Built a house price prediction model using regression. "
                          "Created a weather data pipeline using Python, SQLite and REST APIs.",
            "Education": "B.Tech Computer Science and Engineering, 2026",
        },
    },
    "Priya_Verma.docx": {
        "header": ["Priya Verma", "priya_verma92@gmail.com", "9123456780"],
        "sections": {
            "Summary": "Full stack web developer focused on Java and enterprise applications.",
            "Skills": "Java, Spring Boot, HTML, CSS, JavaScript, React, MySQL, Git, Docker",
            "Experience": "Developed REST APIs using Spring Boot. Built responsive front end with React.",
            "Education": "B.Tech Information Technology, 2025",
        },
    },
    "Amit_Kumar.docx": {
        "header": ["Amit Kumar", "amit.kumar@mail.co.in | (+91) 99887-76655"],
        "sections": {
            "Summary": "Data analyst skilled in dashboards and statistical analysis.",
            "Skills": "Python, SQL, Excel, Power BI, Tableau, Pandas, Statistics, Data Visualization",
            "Experience": "Created business dashboards in Power BI. Cleaned and analysed sales data using Pandas and SQL.",
            "Education": "B.Sc Statistics, 2024",
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