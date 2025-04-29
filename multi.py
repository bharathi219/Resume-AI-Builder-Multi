import streamlit as st
import pdfplumber
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Skills database
SKILLS_DB = [
    'python', 'java', 'sql', 'machine learning', 'deep learning', 'nlp',
    'pandas', 'numpy', 'django', 'flask', 'html', 'css', 'javascript', 'power bi',
    'aws', 'time series forecasting', 'rest api', 'docker', 'fastapi', 'git',
    'linux', 'bash', 'ci/cd', 'postgresql', 'mysql'
]

# Predefined Job Descriptions
JOB_DESCRIPTIONS = {
    "Data Scientist": "machine learning pandas numpy python statistics deep learning",
    "Backend Developer": "django flask python rest api sql postgresql docker",
    "Frontend Developer": "html css javascript react vue responsive design",
    "Data Analyst": "sql excel power bi tableau data visualization python",
    "DevOps Engineer": "aws docker kubernetes ci/cd linux bash cloud",
    "NLP Engineer": "nlp spacy huggingface transformers bert text classification",
    "Python Developer": "flask fastapi python rest api pandas numpy",
    "Software Engineer": "java python git software development lifecycle algorithms",
    "Cloud Engineer": "aws azure cloud docker devops monitoring automation",
    "AI Engineer": "deep learning neural networks tensorflow keras computer vision"
}

# Extract text from PDF resume
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Extract skills from text
def extract_skills(text):
    tokens = [token.text.lower() for token in nlp(text)]
    return sorted(set([skill for skill in SKILLS_DB if skill in tokens]))

# Extract skills from JD
def extract_skills_from_jd(jd_text):
    tokens = [token.text.lower() for token in nlp(jd_text)]
    return sorted(set([skill for skill in SKILLS_DB if skill in tokens]))

# Education extractor
def extract_education(text):
    education = []
    edu_keywords = ['btech', 'mtech', 'b.sc', 'm.sc', 'bachelor', 'master', 'phd']
    for line in text.split('\n'):
        if any(word in line.lower() for word in edu_keywords):
            education.append(line.strip())
    return education

# Experience extractor
def extract_experience(text):
    experience = []
    exp_keywords = ['intern', 'developer', 'engineer', 'analyst', 'consultant']
    for line in text.split('\n'):
        if any(keyword in line.lower() for keyword in exp_keywords):
            experience.append(line.strip())
    return experience

# Project extractor
def extract_projects(text):
    projects = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'project' in line.lower():
            project = ' '.join(lines[i:i+3])
            projects.append(project.strip())
    return projects

# Rank job roles based on cosine similarity
def rank_job_roles(resume_skills, job_descriptions):
    resume_text = ' '.join(resume_skills)
    scores = []
    for role, jd_text in job_descriptions.items():
        jd_skills = extract_skills_from_jd(jd_text)
        jd_text_combined = ' '.join(jd_skills)
        vectorizer = CountVectorizer().fit_transform([resume_text, jd_text_combined])
        vectors = vectorizer.toarray()
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        scores.append((role, round(score, 2)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:10]

# -------------------- Streamlit UI --------------------

st.title("🔍 AI Resume Job Role Matcher")

resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if resume_file:
    # Extract and analyze resume
    resume_text = extract_text_from_pdf(resume_file)
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)
    projects = extract_projects(resume_text)

    # Predict job roles
    top_matches = rank_job_roles(skills, JOB_DESCRIPTIONS)

    # Display top 10 matches first
    if top_matches:
        st.subheader("📊 Top 10 Matching Job Roles:")
        for rank, (role, score) in enumerate(top_matches, start=1):
            st.markdown(f"**{rank}. {role}** — Match Score: `{score * 100:.2f}%`")

        # Display predicted role (top 1)
        predicted_role, predicted_score = top_matches[0]
        st.subheader("🎯 Predicted Job Role:")
        st.success(f"{predicted_role} — Match Score: {predicted_score * 100:.2f}%")
    else:
        st.warning("No matching job roles found.")

    # Display extracted info
    st.subheader("✅ Extracted Skills:")
    st.markdown("• " + "\n• ".join(skills) if skills else "No known skills found.")

    st.subheader("🎓 Education:")
    st.markdown("• " + "\n• ".join(education) if education else "No education info found.")

    st.subheader("💼 Experience:")
    st.markdown("• " + "\n• ".join(experience) if experience else "No experience info found.")

    st.subheader("🛠 Projects:")
    st.markdown("• " + "\n• ".join(projects) if projects else "No project info found.")
