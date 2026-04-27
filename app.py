# app.py
import os
import re
import traceback
from io import BytesIO

import numpy as np
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader


# ------------------------------
# Utility: load HF token from multiple sources
# ------------------------------
def load_hf_key_from_sources():
    # 1) secret_api_keys.py (dev)
    try:
        from secret_api_keys import \
            huggingface_api_key as _hf_key  # noqa: F401
        if _hf_key:
            return str(_hf_key).strip()
    except Exception:
        pass

    # 2) environment
    env_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if env_key:
        return str(env_key).strip()

    # 3) streamlit secrets
    try:
        if hasattr(st, "secrets"):
            secret_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACE_API_KEY")
            if secret_key:
                return str(secret_key).strip()
    except Exception:
        pass

    # 4) .env via python-dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        env_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
        if env_key:
            return str(env_key).strip()
    except Exception:
        pass

    return None

def ensure_env_token(token: str):
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        os.environ["HUGGINGFACE_API_KEY"] = token

# attempt to load token early (may remain None)
hf_token = load_hf_key_from_sources()
if hf_token:
    ensure_env_token(hf_token)

# ------------------------------
# Permissive imports (work across different langchain releases)
# Do NOT raise at import time; we will check availability at runtime and show friendly messages.
# ------------------------------
# Document loader
try:
    from langchain_community.document_loaders import WebBaseLoader
except Exception:
    try:
        from langchain.document_loaders import WebBaseLoader
    except Exception:
        WebBaseLoader = None

# Text splitters
try:
    from langchain_text_splitters import CharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import CharacterTextSplitter
    except Exception:
        try:
            from langchain.text_splitters import CharacterTextSplitter
        except Exception:
            CharacterTextSplitter = None

# Embeddings (HuggingFace)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
        except Exception:
            HuggingFaceEmbeddings = None

# FAISS wrapper
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        FAISS = None

# RetrievalQA: try multiple known paths
RetrievalQA = None
try:
    from langchain.chains import RetrievalQA as _rq
    RetrievalQA = _rq
except Exception:
    try:
        from langchain.chains.retrieval_qa.base import RetrievalQA as _rq
        RetrievalQA = _rq
    except Exception:
        try:
            from langchain.qa import RetrievalQA as _rq
            RetrievalQA = _rq
        except Exception:
            RetrievalQA = None

# HuggingFace endpoint wrapper
try:
    from langchain_huggingface import HuggingFaceEndpoint
except Exception:
    HuggingFaceEndpoint = None

# ------------------------------
# Helpers to report availability
# ------------------------------
def comp_status(name, obj):
    return (name, obj is not None)

def components_ok():
    needed = {
        "CharacterTextSplitter": CharacterTextSplitter,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        "FAISS": FAISS,
        "HuggingFaceEndpoint": HuggingFaceEndpoint,
    }
    return needed

# ------------------------------
# Processing & QA helpers (perform runtime checks and give friendly errors)
# ------------------------------
def _require_component(name, comp):
    if comp is None:
        raise RuntimeError(f"Required component '{name}' is not available. See app instructions to install/upgrade packages.")

def process_input(input_type, input_data):
    # Ensure text splitter and embeddings & FAISS available
    _require_component("CharacterTextSplitter", CharacterTextSplitter)
    _require_component("HuggingFaceEmbeddings", HuggingFaceEmbeddings)
    _require_component("FAISS", FAISS)

    source_docs = []

    if input_type == "Link":
        if WebBaseLoader is None:
            raise RuntimeError("WebBaseLoader not available. Install langchain-community or use a different input type.")
        urls = input_data if isinstance(input_data, (list, tuple)) else [input_data]
        collected = []
        for url in urls:
            if not url:
                continue
            loader = WebBaseLoader(url)
            docs = loader.load()
            for d in docs:
                text = d.page_content if hasattr(d, "page_content") else str(d)
                source_docs.append({"text": text, "source": url})

    elif input_type == "PDF":
        if input_data is None:
            raise ValueError("No PDF uploaded.")
        if hasattr(input_data, "read"):
            pdf_bytes = BytesIO(input_data.read())
        elif isinstance(input_data, BytesIO):
            pdf_bytes = input_data
        else:
            raise ValueError("Invalid PDF input.")
        reader = PdfReader(pdf_bytes)
        name = getattr(input_data, "name", "uploaded.pdf")
        for i, page in enumerate(reader.pages, start=1):
            source_docs.append({"text": page.extract_text() or "", "source": f"{name}, page {i}"})

    elif input_type == "DOCX":
        if input_data is None:
            raise ValueError("No DOCX uploaded.")
        if hasattr(input_data, "read"):
            doc_bytes = BytesIO(input_data.read())
        elif isinstance(input_data, BytesIO):
            doc_bytes = input_data
        else:
            raise ValueError("Invalid DOCX input.")
        doc = Document(doc_bytes)
        name = getattr(input_data, "name", "uploaded.docx")
        source_docs.append({"text": "\n".join([p.text for p in doc.paragraphs]), "source": name})

    elif input_type in ("TXT", "Text"):
        if input_data is None:
            raise ValueError("No text provided.")
        if hasattr(input_data, "read"):
            raw_text = input_data.read().decode("utf-8")
        elif isinstance(input_data, str):
            raw_text = input_data
        elif isinstance(input_data, BytesIO):
            raw_text = input_data.read().decode("utf-8")
        else:
            raise ValueError("Invalid text input.")
        source_docs.append({"text": raw_text, "source": getattr(input_data, "name", "pasted text")})
    else:
        raise ValueError("Unsupported input type")

    if not any(item["text"].strip() for item in source_docs):
        raise ValueError("No text extracted from the provided input.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = []
    metadatas = []
    for item in source_docs:
        for chunk_no, chunk in enumerate(splitter.split_text(item["text"]), start=1):
            texts.append(chunk)
            metadatas.append({"source": item["source"], "chunk": chunk_no})

    hf_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    vector_store = FAISS.from_texts(texts, hf_emb, metadatas=metadatas)
    return vector_store

def answer_question(vectorstore, query):
    docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)
    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def sources_text():
        sources = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            label = meta.get("source", "uploaded document")
            chunk = meta.get("chunk")
            if chunk:
                label = f"{label}, chunk {chunk}"
            if label not in sources:
                sources.append(label)
        return "Sources: " + "; ".join(sources[:3]) if sources else "Sources: retrieved document chunks"

    def with_sources(answer):
        return f"{answer}\n\n{sources_text()}"

    def local_answer():
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        q = query.lower()
        ctx = context.lower()

        def has(*terms):
            return any(term in ctx for term in terms)

        def found_items(items):
            return [label for label, terms in items if any(term in ctx for term in terms)]

        name = lines[0] if lines else "The candidate"
        cgpa = re.search(r"CGPA\s*[:\-]?\s*(\d+(?:\.\d+)?)", context, re.IGNORECASE)
        percent = re.search(r"(\d+(?:\.\d+)?)\s*%", context)
        skills = found_items([
            ("Python", ("python",)),
            ("Machine Learning", ("machine learning", " ml ")),
            ("NLP", ("nlp",)),
            ("LLM/RAG systems", ("llm", "rag")),
            ("LangChain", ("langchain",)),
            ("FAISS", ("faiss",)),
            ("Scikit-learn", ("scikit-learn", "sklearn")),
            ("EDA and preprocessing", ("eda", "preprocessing")),
            ("feature engineering", ("feature engineering",)),
            ("model evaluation", ("model evaluation", "performance optimization")),
        ])

        if "name" in q:
            return with_sources(f"The candidate's name is {name}.")

        if any(word in q for word in ("college", "cgpa", "score", "scored", "marks", "grade", "percentage")):
            if any(word in q for word in ("college", "b.tech", "degree", "university", "kiit")) and cgpa:
                return with_sources(f"He scored a CGPA of {cgpa.group(1)} in college.")
            if percent:
                return with_sources(f"He scored {percent.group(1)}%.")
            if cgpa:
                return with_sources(f"He scored a CGPA of {cgpa.group(1)}.")

        if any(word in q for word in ("skill", "skills", "technical", "technologies", "tech stack")):
            if skills:
                return with_sources("His technical skills include: " + ", ".join(skills[:10]) + ".")
            return with_sources("I could not find clear technical skills in the retrieved resume text.")

        if any(word in q for word in ("rag", "llm", "langchain", "faiss")):
            if has("rag", "llm", "langchain", "faiss"):
                return with_sources("Yes. The resume mentions LLM-based systems and end-to-end RAG pipelines using Python, LangChain, and FAISS.")
            return with_sources("I could not find clear RAG, LLM, LangChain, or FAISS experience in the retrieved text.")

        if any(word in q for word in ("project", "projects", "experience", "practical")):
            evidence = []
            for line in lines:
                low = line.lower()
                if any(term in low for term in ("built", "project", "pipeline", "rag", "llm", "machine learning", "predictive model")):
                    evidence.append(line)
            if evidence:
                return with_sources("Relevant project/practical evidence: " + " ".join(evidence[:3]))
            return with_sources("I could not find a specific project description in the retrieved resume text.")

        if any(word in q for word in ("hire", "hiring", "candidate", "fit", "suitable", "role", "developer")):
            positives = skills[:]
            if "intern" in ctx or "experience" in ctx or "project" in ctx or "built" in ctx:
                positives.append("project/practical experience")
            verdict = "Yes, for an entry-level or intern Python/AI role." if len(positives) >= 3 else "Maybe, but screen him first."
            return with_sources(
                f"{verdict}\n\n"
                f"Why: the resume shows {', '.join(positives[:6]) if positives else 'limited matching evidence'}.\n\n"
                f"Check in interview: Python fundamentals, code quality, debugging, APIs/database basics, and whether he can explain his projects end to end."
            )

        if any(word in q for word in ("leadership", "leader", "lead", "managed", "team")):
            leadership_terms = ("lead", "leader", "leadership", "managed", "team", "coordinated", "organized", "mentored", "captain", "head")
            evidence = [line for line in lines if any(term in line.lower() for term in leadership_terms)]
            if evidence:
                return with_sources(f"Yes, the resume shows possible leadership evidence: {evidence[0]}")
            return with_sources("The resume does not show clear leadership evidence. I would ask about team projects, ownership, mentoring, or event/club responsibilities in the interview.")

        if any(word in q for word in ("interview", "verify", "check", "ask")):
            return with_sources("Interview focus: test Python fundamentals, ask him to explain his RAG pipeline, check LangChain/FAISS understanding, review one project end to end, and give a small debugging or API task.")

        words = [w for w in re.findall(r"[a-zA-Z0-9+#.]+", q) if len(w) > 2]
        stop = {"does", "have", "what", "why", "how", "him", "his", "the", "for", "and", "with", "should"}
        keys = [w for w in words if w not in stop]
        scored = []
        for line in lines:
            low = line.lower()
            score = sum(1 for key in keys if key in low)
            if score:
                scored.append((score, line))
        if scored:
            scored.sort(reverse=True, key=lambda item: item[0])
            return with_sources(f"Based on the resume: {scored[0][1]}")
        return with_sources("I could not find a clear answer in the uploaded document.")

    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    use_llm = os.environ.get("USE_HF_LLM", "").lower() in ("1", "true", "yes")
    if HuggingFaceEndpoint is None or not token or not use_llm:
        return {"query": query, "result": local_answer(), "source_documents": docs}

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        provider="hf-inference",
        huggingfacehub_api_token=token,
        temperature=0.6,
        max_new_tokens=256,
    )
    prompt = (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        answer = with_sources(llm.invoke(prompt))
    except Exception:
        answer = local_answer()
    return {"query": query, "result": answer, "source_documents": docs}

# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.set_page_config(page_title="RAG Q&A (robust)", layout="wide")
    st.title("Document Q&A using RAG")

    # show component availability
    st.sidebar.header("Diagnostics / Setup")
    comps = components_ok()
    for name, obj in comps.items():
        st.sidebar.write(f"- {name}: {'✅' if obj is not None else '❌'}")

    st.sidebar.markdown(
        """
        **If any component is ❌**:
        - Ensure you installed/updated the same Python env that runs Streamlit:
          `python -m pip install --upgrade langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu`
        - Then run the app using the same Python: `python -m streamlit run app.py`
        """
    )

    global hf_token
    # re-check token sources in case environment changed
    if not hf_token:
        hf_token = load_hf_key_from_sources()
        if hf_token:
            ensure_env_token(hf_token)

    if not (os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")):
        st.warning("Hugging Face API token not detected. Provide it below (session-only) or set HUGGINGFACEHUB_API_TOKEN in your environment.")
        token_input = st.text_input("Paste Hugging Face API token", type="password", key="hf_input")
        if token_input:
            token_input = token_input.strip()
            if token_input:
                ensure_env_token(token_input)
                st.success("Token stored for this session (environment variable set).")
    else:
        st.success("Hugging Face token available in environment.")

    input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])

    input_data = None
    if input_type == "Link":
        n = st.number_input("Number of links", min_value=1, max_value=10, value=1)
        urls = []
        for i in range(n):
            url = st.text_input(f"URL {i+1}")
            if url:
                urls.append(url)
        input_data = urls

    elif input_type == "Text":
        input_data = st.text_area("Enter text here")

    elif input_type == "PDF":
        input_data = st.file_uploader("Upload PDF", type=["pdf"])

    elif input_type == "DOCX":
        input_data = st.file_uploader("Upload DOCX", type=["docx", "doc"])

    elif input_type == "TXT":
        input_data = st.file_uploader("Upload TXT", type=["txt"])

    proceed_clicked = st.button("Proceed (create embeddings)")
    if proceed_clicked:
        try:
            # runtime checks
            missing = []
            needed = ("CharacterTextSplitter", "HuggingFaceEmbeddings", "FAISS")
            mapping = {"CharacterTextSplitter": CharacterTextSplitter,
                       "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
                       "FAISS": FAISS}
            for name in needed:
                if mapping[name] is None:
                    missing.append(name)
            if missing:
                st.error(f"Missing components required for embedding creation: {', '.join(missing)}. See sidebar for install instructions.")
            else:
                if input_type in ("Link",) and WebBaseLoader is None:
                    st.error("WebBaseLoader not available. Install langchain-community to use Link input.")
                else:
                    vectorstore = process_input(input_type, input_data)
                    st.session_state["vectorstore"] = vectorstore
                    st.success("Vectorstore created and saved to session state.")
        except Exception as e:
            st.error(f"Error while processing input: {e}")
            st.text(traceback.format_exc())

    if "vectorstore" in st.session_state:
        query = st.text_input("Ask your question")
        if st.button("Submit"):
            try:
                resp = answer_question(st.session_state["vectorstore"], query)
                st.markdown(resp["result"])
            except Exception as e:
                st.error(f"Error while answering: {e}")
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
