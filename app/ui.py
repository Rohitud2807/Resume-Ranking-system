import streamlit as st
import tempfile
import os

from nlp_engine import(
    extract_text_from_pdf,
    clean_text,
    vectorize_text,
    calculate_cosine_similarity,
    extract_skills,
    load_skills
)
st.set_page_config(page_title="Resume Ranking System" , layout="centered")
st.title("Resume Screening & Ranking System")
st.write("Upload resumes and compare them against a job description.")

#upload Resumes
uploaded_resumes = st.file_uploader("Upload Resume pdfs",
                                    type=["pdf"],
                                    accept_multiple_files=True)

#Jd input
jd_text = st.text_area("paste job description here")

#load skills
skills = load_skills("data/skills.csv")

if st.button("Rank Resumes"):
    if not uploaded_resumes or not jd_text:
        st.warning("Please upload resumes and enter a job description.")
        st.stop()
    cleaned_jd = clean_text(jd_text)
    jd_skills = extract_skills(cleaned_jd,skills)

    results = []
    for resume in uploaded_resumes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resume.read())
            pdf_path = tmp.name

        resume_text = clean_text(extract_text_from_pdf(pdf_path))

        vectors = vectorize_text(resume_text, cleaned_jd)
        tfidf_score = calculate_cosine_similarity(vectors) * 100

        resume_skills = extract_skills(resume_text, skills)
        matched = list(set(resume_skills) & set(jd_skills))
        skill_score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

        final_score = (0.7 * tfidf_score) + (0.3 * skill_score)

        results.append({
            "name": resume.name,
            "score": round(final_score, 2),
            "matched_skills": matched
        })

        os.remove(pdf_path)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    st.subheader("📊 Ranking Results")
    for i, r in enumerate(results, 1):
        st.write(f"**{i}. {r['name']}** — {r['score']}%")
        st.caption(f"Matched skills: {', '.join(r['matched_skills'])}")