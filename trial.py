#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from flask import Flask, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename  # (kept in case your templates still use it)

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from PyPDF2 import PdfReader

# =========================== CONFIG ===================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
FAISS_DIR = BASE_DIR / "faiss_index"
INDEX_NAME = "policy_index"

FAISS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

LLM_MODEL = "gemini-2.5-flash"

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

# =========================== UTILITIES ===================================

_embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
_vectorstore: Optional[FAISS] = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return _embeddings


def build_or_load_vectorstore() -> FAISS:
    """
    Creates or loads a FAISS vector store from PDFs in the documents directory.
    Uses index_name to avoid dangerous deserialization flags.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()

    index_path = FAISS_DIR / INDEX_NAME
    if not index_path.exists():
        print(f"No FAISS index found. Building one from PDFs in '{DOCS_DIR}'...")
        loader = PyPDFDirectoryLoader(str(DOCS_DIR), recursive=True)
        documents: List[Document] = loader.load()
        if not documents:
            raise RuntimeError(
                f"No PDF documents found in '{DOCS_DIR}'. Please add policy documents to this folder."
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(FAISS_DIR), index_name=INDEX_NAME)
        print("FAISS index built and saved.")
    else:
        print("Loading existing FAISS index.")

    _vectorstore = FAISS.load_local(
        str(FAISS_DIR), embeddings, index_name=INDEX_NAME
    )
    return _vectorstore


def get_context_from_vectorstore(query: str, k: int = 4) -> str:
    db = build_or_load_vectorstore()
    docs = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def get_pdf_text(file_storage: Optional[FileStorage]) -> str:
    if not file_storage or not file_storage.filename:
        return ""
    try:
        pdf_reader = PdfReader(file_storage)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error reading PDF {file_storage.filename}: {e}")
        return ""


def get_llm(model: str = LLM_MODEL) -> genai.GenerativeModel:
    # Using kwargs form is forward compatible with google-generativeai>=0.7
    return genai.GenerativeModel(
        model_name=model,
        generation_config={
            "temperature": 0.2,
        },
    )


def extract_bill_info(bill_text: str) -> Dict[str, Any]:
    """
    Uses Gemini in JSON mode to extract 'disease' and 'expense' from bill text.
    Returns a dict with 'disease' and 'expense'.
    """
    if not bill_text.strip():
        return {"disease": None, "expense": None}

    system_prompt = (
        "You are an expert at extracting structured information from raw text. "
        "From the following medical invoice details, extract the patient's primary 'disease' "
        "and the total 'expense' amount. Respond with only a valid JSON object like: "
        '{"disease": "...", "expense": 12345.67}'
    )

    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    try:
        response = model.generate_content(
            [system_prompt, f"INVOICE DETAILS:\n{bill_text}"]
        )
        # response.text is still supported; if it changes in future, use candidates[0].content.parts
        data = json.loads(response.text)
        # Normalize keys/types
        disease = data.get("disease")
        expense = data.get("expense")
        try:
            expense = float(expense) if expense is not None else None
        except (TypeError, ValueError):
            expense = None
        return {"disease": disease, "expense": expense}
    except Exception as e:
        print(f"Error during bill information extraction: {e}")
        return {"disease": None, "expense": None, "error": str(e)}


# =========================== FLASK ROUTES ===================================

@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/")
def process_claim() -> str:
    try:
        # 1) Form data
        form_data = request.form
        render_ctx = {key: form_data.get(key, "").strip() for key in form_data}
        medical_bill_file: Optional[FileStorage] = request.files.get("medical_bill")

        # 2) Validation
        if not all(
            [
                render_ctx.get("name"),
                render_ctx.get("claim_type"),
                render_ctx.get("total_claim_amount"),
            ]
        ):
            render_ctx[
                "output"
            ] = "Validation Error: Name, Claim Type, and Claim Amount are required."
            return render_template("result.html", **render_ctx)

        # 3) Extract bill info
        bill_text = get_pdf_text(medical_bill_file)
        bill_info = extract_bill_info(bill_text)

        # 4) Amount mismatch validation (cheap pre-check)
        try:
            claimed_amount = float(render_ctx["total_claim_amount"])
            billed_expense = bill_info.get("expense") or 0.0
            if billed_expense > 0 and claimed_amount > billed_expense:
                render_ctx["output"] = (
                    f"Claim Rejected: The claimed amount (${claimed_amount:,.2f}) "
                    f"is greater than the amount on the medical bill (${billed_expense:,.2f})."
                )
                return render_template("result.html", **render_ctx)
        except (ValueError, TypeError):
            # Ignore; let LLM reason about it
            pass

        # 5) Build LLM prompt
        patient_info = "\n".join(
            f"- {key.replace('_', ' ').title()}: {value}"
            for key, value in render_ctx.items()
            if value
        )
        medical_bill_info = (
            f"Disease mentioned in bill: {bill_info.get('disease')}\n"
            f"Full Bill Text: {bill_text}"
        )

        final_prompt = prompt.format(
            claim_approval_context=get_context_from_vectorstore(
                "documents required for claim approval"
            ),
            general_exclusion_context=get_context_from_vectorstore(
                "list of all general exclusions"
            ),
            patient_info=patient_info,
            medical_bill_info=medical_bill_info,
            max_amount=render_ctx.get("total_claim_amount"),
        )

        # 6) Generate report
        model = get_llm()
        response = model.generate_content(final_prompt)
        output = (response.text or "").replace("\n", "<br>")

        render_ctx["output"] = output or "No response produced by the model."
        return render_template("result.html", **render_ctx)

    except Exception as e:
        print(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
        return render_template("result.html", output=f"An unexpected server error occurred: {e}")


# =========================== MAIN ===================================

if __name__ == "__main__":
    # Warm up vector store to avoid first-request latency
    build_or_load_vectorstore()
    app.run(host="0.0.0.0", port=8081, debug=True)
