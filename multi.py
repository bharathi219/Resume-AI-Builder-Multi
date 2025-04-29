import streamlit as st
import pdfplumber
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

SKILLS_DB = ['python', 'java', 'sql', 'machine learning', 'deep learning', 'nlp', 'pandas', 'numpy', 'django', 'flask', 'html', 'css',
             'javascript', 'power bi', 'aws', 'time series forecasting', 'rest api', 'docker', 'fastapi', 'git', 'linux', 'bash', 'ci/cd', 'postgresql', 'mysql']

# Predefined Job Role Database with representative skills (you can customize this)
JOB_ROLES = {
    "Data Scientist": "machine learning pandas numpy python statistics deep learning",
    "Data Analyst": "sql excel tableau power bi python statistics visualization",
    "Backend Developer": "python django flask rest api postgresql mysql",
    "Frontend Developer": "html css javascript react angular ui design",
    "DevOps Engineer": "aws docker ci/cd kubernetes linux bash cloud",
    "NLP Engineer": "nlp spacy bert transformers named entity recognition",
    "Python Developer": "python flask fastapi oop rest api postgresql",
    "Full Stack Developer": "html css javascript django flask react sql",
    "ML Engineer": "machine learning deep learning keras tensorflow scikit-learn",
    "Cloud Engineer": "aws azure gcp cloud computing kubernetes deployment"
}

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_skills(text):
    tokens = [token.text.lower() for token in nlp(text)]
    return sorted(set([skill for skill in SKILLS_DB if skill in tokens]))
 
def extract_skills_from_jd(jd_text):
    tokens = [token.text.lower() for token in nlp(jd_text)]
    return sorted(set([skill for skill in SKILLS_DB if skill in tokens]))

def extract_education(text):
    education = []
    edu_keywords = ['btech', 'mtech', 'b.sc', 'm.sc', 'bachelor', 'master', 'phd']
    for line in text.split('\n'):
        if any(word in line.lower() for word in edu_keywords):
            education.append(line.strip())
    return education

def extract_experience(text):
    experience = []
    exp_keywords = ['intern', 'developer', 'engineer', 'analyst', 'consultant']
    for line in text.split('\n'):
        if any(keyword in line.lower() for keyword in exp_keywords):
            experience.append(line.strip())
    return experience

def extract_projects(text):
    projects = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'project' in line.lower():
            project = ' '.join(lines[i:i+3])
            projects.append(project.strip())
    return projects

def rank_job_roles(resume_skills, job_roles_dict):
    resume_str = ' '.join(resume_skills)
    results = []
    for role, skills_str in job_roles_dict.items():
        vectorizer = CountVectorizer().fit_transform([skills_str, resume_str])
        vectors = vectorizer.toarray()
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        results.append((role, round(score * 100, 2)))  # score in percentage
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]


# Streamlit UI
st.title("🔍 AI Top 10 Job Role Predictor from Resume")

resume_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

if resume_file:
    resume_text = extract_text_from_pdf(resume_file)
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)
    projects = extract_projects(resume_text)
    top_roles = rank_job_roles(skills, JOB_ROLES)

    st.subheader("🎯 Top 10 Predicted Job Roles:")
    for idx, (role, score) in enumerate(top_roles, start=1):
        st.markdown(f"**{idx}. {role}** — Score: `{score:.2f}%`")

    st.subheader("✅ Extracted Skills:")
    if skills:
        st.markdown("• " + "\n• ".join(skills))
    else:
        st.warning("No known skills detected.")

    st.subheader("🎓 Education:")
    if education:
        st.markdown("• " + "\n• ".join(education))
    else:
        st.warning("No education details found.")

    st.subheader("💼 Experience:")
    if experience:
        st.markdown("• " + "\n• ".join(experience))
    else:
        st.warning("No experience entries detected.")

    st.subheader("🛠 Projects:")
    if projects:
        st.markdown("• " + "\n• ".join(projects))
    else:
        st.warning("No project information found.")

