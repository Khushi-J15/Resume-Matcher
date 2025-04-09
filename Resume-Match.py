import streamlit as st
import pandas as pd
import os
import docx2txt
import PyPDF2
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Config ---
FOLDER = "input_data/"
PICKLE_FILE = "resume_vectors4.pkl"

# Function to extract text from files
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# UI
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="centered"
)

st.title("📄 SkillMap AI")

st.sidebar.header("Resume Weight Configuration")
skill_weight = st.sidebar.slider("Weight for Skills", 1, 5, 3)
experience_weight = st.sidebar.slider("Weight for Experience", 1, 5, 3)

# Load and process resumes from folder
folder_resumes, folder_files, folder_buffers = [], [], []

if os.path.exists(FOLDER):
    for file in os.listdir(FOLDER):
        path = os.path.join(FOLDER, file)
        if os.path.isfile(path):
            text = extract_text(path)
            if text:
                folder_resumes.append(text)
                folder_files.append(file)

                buffer = BytesIO()
                buffer.write(text.encode("utf-8"))
                buffer.seek(0)
                folder_buffers.append(buffer)

#  TF-IDF Vectorization
if not os.path.exists(PICKLE_FILE):
    vectorizer = TfidfVectorizer(stop_words="english")
    resume_vectors = vectorizer.fit_transform(folder_resumes)

    with open(PICKLE_FILE, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "vectors": resume_vectors,
            "files": folder_files,
            "resumes": folder_resumes,
            "buffers": folder_buffers
        }, f)
else:
    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)
        vectorizer = data["vectorizer"]
        resume_vectors = data["vectors"]
        folder_files = data["files"]
        folder_resumes = data["resumes"]
        folder_buffers = data["buffers"]

# Input Job Description
job_description = st.text_area("📾 Enter Job Description (include key skills & experience):")

if st.button("🔍 Match Resumes"):
    if job_description.strip() == "":
        st.warning("⚠️ Please enter a job description.")
    else:
        boosted_job = (job_description + " ") * (skill_weight + experience_weight)
        job_vector = vectorizer.transform([boosted_job])

        similarities = cosine_similarity(job_vector, resume_vectors)[0]
        top_indices = similarities.argsort()[::-1]

        # Get top 5 with similarity > 0
        top_matches = [(i, round(similarities[i] * 100, 2)) for i in top_indices if similarities[i] > 0][:5]

        if not top_matches:
            st.error("No matching resumes found for the given job description.")
        else:
            st.subheader("🌟 Top 5 Matching Resumes")

            similarity_data2 = {
                "Name": [folder_files[i] for i, _ in top_matches],
                "Similarity": [score for _, score in top_matches]
            }
            df_sim = pd.DataFrame(similarity_data2)
            with open("similarity_data4.pkl", "wb") as f:
                pickle.dump(df_sim, f)

            for i, score in top_matches:
                file = folder_files[i]
                text = folder_resumes[i]
                buffer = folder_buffers[i]

                with st.expander(f"📁 {file} — Match Score: {score}%"):
                    st.write(text[:800])
                    st.download_button("📅 Download Resume",
                                       data=buffer,
                                       file_name=file,
                                       mime="text/plain")
