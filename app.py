#!/usr/bin/env python3
import os
import json
import re
from pathlib import Path
from typing import Sequence, Dict, Any

from dotenv import load_dotenv
from flask import Flask, render_template, request, Response
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# ----------------------------- CONFIG ---------------------------------
# Load environment variables from .env file
load_dotenv()

# --- API Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

# Set the key for all Google AI/ML libraries
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

# --- Paths ---
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
FAISS_DIR = BASE_DIR / "faiss_index"
FAISS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --- Flask ---
app = Flask(__name__)

# --- Model & Prompt Configuration ---
LLM_MODEL = "gemini-2.5-flash"

# This single, comprehensive prompt is given to the LLM.
# It contains all necessary context and instructions, empowering the model
# to perform all checks, including document verification and exclusion analysis.
PROMPT_TEMPLATE = """You are an expert AI assistant for verifying health insurance claims. Your task is to analyze the provided information and determine if the claim should be approved or rejected. Follow these steps carefully:

1. **Exclusion Check**: Review the patient's medical information and compare it against the general exclusions list from the context. If the claimed disease or billed diagnosis matches any exclusion, set `EXCLUSION_FOUND` to `TRUE`. Otherwise, set it to `FALSE`.

2. **Mismatch Checks**:
   - If the patient name in the claim differs from the name in the medical bill, REJECT.
   - If the claimed amount differs from the billed amount, REJECT.
   - If the claimed disease/reason differs from the billed diagnosis, REJECT.

3. **Final Decision**:
    * **REJECT** the claim if `EXCLUSION_FOUND` is `TRUE`.
    * **REJECT** the claim if any mismatch checks fail.
    * Otherwise, **ACCEPT** the claim.

**Reference Context: General Exclusions List**
{general_exclusion_context}

**Patient and Claim Information**
{patient_info}

**Medical Bill Information**
{medical_bill_info}

Based on your analysis, generate a detailed report in HTML-friendly text. The report must include:

- A final verdict: **Claim Accepted** or **Claim Rejected**.
- If accepted, state the maximum approved amount: {max_amount}.
- The status of the exclusion check: `EXCLUSION_FOUND: [TRUE/FALSE]`.
- Clear reasons for rejection (mismatches or exclusions).
- A structured report with the following sections: Executive Summary, Introduction, Claim Details, Exclusion and Mismatch Verification, and Fraud Check.
"""

prompt = PromptTemplate(
    input_variables=[
        "claim_approval_context",
        "general_exclusion_context",
        "patient_info",
        "medical_bill_info",
        "max_amount",
    ],
    template=PROMPT_TEMPLATE,
)


# --------------------------- UTILITIES ---------------------------------

def get_vectorstore() -> FAISS:
    """Creates a FAISS vector store from PDFs in the documents directory, caching it for reuse."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    if not any(FAISS_DIR.iterdir()):
        print(f"No FAISS index found. Building one from PDFs in '{DOCS_DIR}'...")
        loader = DirectoryLoader(
            str(DOCS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        documents = loader.load()
        if not documents:
            raise RuntimeError(f"No PDF documents found in '{DOCS_DIR}'. Please add policy documents to this folder.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(FAISS_DIR))
        print("FAISS index built and saved.")
    else:
        print("Loading existing FAISS index.")

    return FAISS.load_local(
        str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
    )


def get_context_from_vectorstore(query: str, k: int = 4) -> str:
    """Retrieves relevant context from the vector store for a given query."""
    db = get_vectorstore()
    docs = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def get_pdf_text(file_storage: FileStorage | None) -> str:
    """Extracts text content from an uploaded PDF file."""
    if not file_storage or not file_storage.filename:
        return ""
    try:
        pdf_reader = PdfReader(file_storage)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error reading PDF {file_storage.filename}: {e}")
        return ""


def extract_bill_info(bill_text: str) -> Dict[str, Any]:
    """Uses the Gemini model in JSON mode to extract 'disease' and 'expense' from bill text."""
    if not bill_text.strip():
        return {"disease": None, "expense": None}

    system_prompt = (
        "You are an expert at extracting structured information from raw text. "
        "From the following medical invoice details, extract the patient's primary 'disease' "
        "and the total 'expense' amount. Respond with only a valid JSON object like: "
        '{"disease": "...", "expense": 12345.67}'
    )

    try:
        model = genai.GenerativeModel(
            LLM_MODEL,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        response = model.generate_content([system_prompt, f"INVOICE DETAILS:\n{bill_text}"])
        return json.loads(response.text)
    except Exception as e:
        print(f"Error during bill information extraction: {e}")
        return {"disease": None, "expense": None, "error": str(e)}


# ------------------------------ FLASK ROUTES ---------------------------------

@app.get("/")
def index() -> str:
    """Renders the main page with the claim submission form."""
    return render_template("index.html")


@app.post("/")
def process_claim() -> str:
    """Processes the submitted claim form and returns the adjudication result."""
    try:
        # --- 1. Get Form Data & Prepare Context ---
        form_data = request.form
        render_ctx = {key: form_data.get(key, "").strip() for key in form_data}
        medical_bill_file = request.files.get("medical_bill")
        
        # --- 2. Initial Validation ---
        if not all([render_ctx.get("name"), render_ctx.get("claim_type"), render_ctx.get("total_claim_amount")]):
            render_ctx["output"] = "Validation Error: Name, Claim Type, and Claim Amount are required."
            return render_template("result.html", **render_ctx)

        # --- 3. Extract Info from Medical Bill ---
        bill_text = get_pdf_text(medical_bill_file)
        bill_info = extract_bill_info(bill_text)
        
        # --- 4. Amount Mismatch Validation ---
        try:
            claimed_amount = float(render_ctx["total_claim_amount"])
            billed_expense = float(bill_info.get("expense", 0))
            if billed_expense > 0 and claimed_amount > billed_expense:
                render_ctx["output"] = (
                    f"Claim Rejected: The claimed amount (${claimed_amount:,.2f}) "
                    f"is greater than the amount on the medical bill (${billed_expense:,.2f})."
                )
                return render_template("result.html", **render_ctx)
        except (ValueError, TypeError):
            # Could not parse numbers, let the LLM handle it.
            pass

        # --- 5. Build the Prompt for the LLM ---
        patient_info = "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in render_ctx.items() if value)
        medical_bill_info = f"Disease mentioned in bill: {bill_info.get('disease')}\nFull Bill Text: {bill_text}"

        final_prompt = prompt.format(
            claim_approval_context=get_context_from_vectorstore("documents required for claim approval"),
            general_exclusion_context=get_context_from_vectorstore("list of all general exclusions"),
            patient_info=patient_info,
            medical_bill_info=medical_bill_info,
            max_amount=render_ctx.get("total_claim_amount"),
        )
        
        # --- 6. Generate Report with LLM ---
        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(final_prompt)
        
        # Format for HTML display and render
        output = response.text.replace("\n", "<br>")
        render_ctx["output"] = output
        return render_template("result.html", **render_ctx)

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
        return render_template("result.html", output=f"An unexpected server error occurred: {e}")


# ------------------------------ MAIN -----------------------------------

# if __name__ == "__main__":
#     # Initialize the vector store on startup to avoid delays on the first request.
#     get_vectorstore()
#     app.run(host="0.0.0.0", port=8081, debug=True)
