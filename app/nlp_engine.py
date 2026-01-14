import os
import re
import pdfplumber
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()


def load_job_description(jd_path):
    with open(jd_path, "r", encoding="utf-8") as file:
        return file.read()


def vectorize_text(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    return vectors


def calculate_cosine_similarity(vectors):
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]


def extract_skills(text, skills_list):
    return [skill for skill in skills_list if skill in text]


def load_skills(skill_path):
    df = pd.read_csv(skill_path)
    return df["skill"].str.lower().tolist()

if __name__ == "__main__":

    # Load & clean Job Description
    jd_path = "data/job_descriptions/jd1.txt"
    jd_text = load_job_description(jd_path)
    cleaned_jd = clean_text(jd_text)

    # Load skills and JD skills
    skills = load_skills("data/skills.csv")
    jd_skills = extract_skills(cleaned_jd, skills)

    results = []

    # Loop through all resumes
    for file in os.listdir("data/sample_resumes"):
        if file.endswith(".pdf"):
            pdf_path = f"data/sample_resumes/{file}"

            resume_text = clean_text(extract_text_from_pdf(pdf_path))

            # TF-IDF similarity
            vectors = vectorize_text(resume_text, cleaned_jd)
            similarity_score = calculate_cosine_similarity(vectors) * 100

            # Skill matching
            resume_skills = extract_skills(resume_text, skills)
            matched_skills = list(set(resume_skills) & set(jd_skills))
            skills_score = (
                len(matched_skills) / len(jd_skills) * 100
                if jd_skills else 0
            )

            # Final weighted score 
            final_score = (0.7 * similarity_score) + (0.3 * skills_score)

            results.append({
                "resume": file,
                "final_score": round(final_score, 2)
            })

    # Rank resumes
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    print("\n--- Resume Ranking ---")
    for i, res in enumerate(results, start=1):
        print(f"{i}. {res['resume']} → {res['final_score']}%")
