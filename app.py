from flask import Flask, render_template, request
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

SKILLS = [
    "python", "java", "c", "c++", "html", "css", "javascript",
    "react", "node", "flask", "django", "sql", "mysql", "mongodb",
    "machine learning", "deep learning", "data analysis", "pandas",
    "numpy", "git", "github", "api", "rest api", "firebase",
    "android", "kotlin", "problem solving", "communication"
]

def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text.lower()

def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))

def calculate_match_score(resume_text, job_desc):
    data = [resume_text, job_desc]

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(data)

    score = cosine_similarity(matrix)[0][1]
    return round(score * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        resume = request.files["resume"]
        job_desc = request.form["jobdesc"].lower()

        resume_text = extract_text_from_pdf(resume)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_desc)

        missing_skills = list(set(job_skills) - set(resume_skills))

        match_score = calculate_match_score(resume_text, job_desc)

        if match_score >= 75:
            status = "Excellent Match"
        elif match_score >= 50:
            status = "Good Match"
        else:
            status = "Needs Improvement"

        result = {
            "score": match_score,
            "status": status,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "missing_skills": missing_skills
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)