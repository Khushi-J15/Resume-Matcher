from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

DATASET_FOLDER = 'input_data/'  # Folder where your dataset of resumes is stored
app.config['UPLOAD_FOLDER'] = 'uploads/'  # For user uploads (optional)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Skill set for parsing
SKILL_SET = [
    "python", "java", "c++", "unity", "streamlit", "opencv",
    "react", "firebase", "git", "machine learning", "nlp"
]

# ---------- Resume Parsing (spaCy-based) ----------
def parse_resume_spacy(text):
    parsed = {}
    doc = nlp(text)

    # Name (first PERSON entity)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    parsed["name"] = name

    # Email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    parsed["email"] = email_match.group(0) if email_match else None

    # Phone
    phone_match = re.search(r"\+?\d[\d -]{8,13}\d", text)
    parsed["phone"] = phone_match.group(0) if phone_match else None

    # Skills
    found_skills = set()
    for token in doc:
        if token.text.lower() in SKILL_SET:
            found_skills.add(token.text.lower())
    parsed["skills"] = list(found_skills)

    return parsed

# ---------- Text Extraction ----------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

# ---------- Load and Vectorize Resumes ----------
resume_filenames = []
resume_texts = []

for filename in os.listdir(DATASET_FOLDER):
    path = os.path.join(DATASET_FOLDER, filename)
    if os.path.isfile(path):
        resume_filenames.append(filename)
        resume_texts.append(extract_text(path))

vectorizer = TfidfVectorizer().fit(resume_texts)
resume_vectors = vectorizer.transform(resume_texts)

# ---------- Flask Routes ----------
@app.route("/")
def home():
    return render_template('matchresume.html')

@app.route("/matcher", methods=["POST"])
def matcher():
    job_desc = request.form['job_description']

    if not job_desc.strip():
        return render_template('matchresume.html', message="Please enter a job description.")

    job_vector = vectorizer.transform([job_desc])
    similarities = cosine_similarity(job_vector, resume_vectors)[0]

    top_indices = similarities.argsort()[-5:][::-1]
    top_resumes = [resume_filenames[i] for i in top_indices]
    similarity_scores = [round(similarities[i] * 100) for i in top_indices]

    matched_resumes = []

    for i in top_indices:
        parsed = parse_resume_spacy(resume_texts[i])
        matched_resumes.append({
            "filename": resume_filenames[i],
            "score": round(similarities[i] * 100),
            "name": parsed.get("name"),
            "email": parsed.get("email"),
            "phone": parsed.get("phone"),
            "skills": ", ".join(parsed.get("skills", []))
        })

    return render_template('matchresume.html',
                           message="Top matching resumes from dataset:",
                           matched_resumes=matched_resumes)

# ---------- Run App ----------
if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
