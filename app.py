import io

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

try:
    import docx
except ImportError:
    docx = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

@st.cache_resource
def get_llm():
    return ChatOllama(model="qwen2.5:1.5b", temperature=0)

llm = get_llm()

prompt = PromptTemplate(
    input_variables=["resume", "jd"],
    template="""
You are an expert HR analyst.

Given this Resume:
{resume}

And this Job Description:
{jd}

Please provide:
1. A match score out of 100
2. Top 5 matching skills/experiences
3. Top 5 missing skills or gaps
4. 3 specific suggestions to improve the resume for this JD

Be concise and structured.
"""
)

st.title("Resume–JD Matcher")
st.subheader("Upload your Resume document or paste it below")

resume_file = st.file_uploader(
    "Upload Resume document",
    type=["txt", "pdf", "docx"],
    help="Supported formats: .txt, .pdf, .docx"
)

resume_text = ""

if resume_file is not None:
    def parse_resume_file(uploaded_file):
        file_name = uploaded_file.name.lower()
        file_bytes = uploaded_file.read()

        if file_name.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="ignore")

        if file_name.endswith(".docx"):
            if docx is None:
                st.error("python-docx is required to parse DOCX resumes. Please install it.")
                return ""
            document = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in document.paragraphs if p.text)

        if file_name.endswith(".pdf"):
            if PdfReader is None:
                st.error("PyPDF2 is required to parse PDF resumes. Please install it.")
                return ""
            reader = PdfReader(io.BytesIO(file_bytes))
            text = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(text)

        return file_bytes.decode("utf-8", errors="ignore")

    resume_text = parse_resume_file(resume_file)

    if resume_text:
        st.success(f"Loaded resume from {resume_file.name}")
    else:
        st.warning("Could not parse the uploaded resume. Please paste the resume text instead.")

if not resume_text:
    resume_text = st.text_area("Or paste your Resume", height=200)

jd_text = st.text_area("Job Description", height=200)

if st.button("Analyze Match"):
    if resume_text and jd_text:
        with st.spinner("Analyzing... (local model may take 30-60 seconds)"):
            chain = prompt | llm
            result = chain.invoke({
                "resume": resume_text,
                "jd": jd_text
            })
            st.markdown(result.content)
    else:
        st.warning("Please provide a resume (uploaded or pasted) and a job description.")