from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import os
import io
from typing import List, Dict
import logging
from sentence_transformers import SentenceTransformer, util
import json  # ✅ Make sure this line is present



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

llm = OllamaLLM(model="llama3.2:1b", temperature=0.5)
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")  # Semantic search model
response_cache = {}

def process_pdf(uploaded_file: UploadFile) -> str:
    """Extract text from PDF and create FAISS embeddings."""
    try:
        pdf_bytes = uploaded_file.file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        pdf_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])

        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        db_path = os.path.join(VECTOR_STORE_DIR, uploaded_file.filename[:-4])
        os.makedirs(db_path, exist_ok=True)

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_text(pdf_text)  # Extract chunks

        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        db = FAISS.from_texts(texts, embeddings)
        db.save_local(db_path)

        # ✅ Ensure metadata correctly saves chunk count
        metadata_file = os.path.join(db_path, "metadata.json")
        metadata = {
            "chunks": len(texts),  # Store actual chunk count
            "status": "Embeddings exist",
            "pdf_name": uploaded_file.filename
        }

        # Debugging step: print chunk count to console
        print(f"Saving metadata for {uploaded_file.filename} with {len(texts)} chunks")

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"FAISS index saved to: {db_path} with {len(texts)} chunks")
        return uploaded_file.filename

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")



@app.post("/upload/")
async def upload_pdf(files: List[UploadFile] = File(...)) -> Dict[str, str]:
    """Handles PDF uploads, processes them, and saves embeddings."""
    results = {}
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        pdf_name = process_pdf(file)
        results[pdf_name] = "Successfully processed and saved FAISS index."

    return results

@app.post("/ask/")
async def ask_question(question: str = Form(...), pdf_name: str = Form(...)):
    """Retrieve the most relevant answer for a user's question using FAISS."""
    cache_key = f"{pdf_name}_{question}"
    if cache_key in response_cache:
        return {"response": response_cache[cache_key]}

    db_path = os.path.join(VECTOR_STORE_DIR, pdf_name[:-4])
    if not os.path.exists(db_path):
        raise HTTPException(status_code=400, detail=f"PDF '{pdf_name}' not found. Please upload it first.")

    try:
        # Load FAISS index
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # 🔹 Retrieve top 5 most relevant chunks for better accuracy
        retriever = db.as_retriever(search_kwargs={"k": 3})
        retrieved_chunks = retriever.get_relevant_documents(question)

        if not retrieved_chunks:
            return {"response": "No relevant information found in the PDF."}

        # **Step 1: Compute Semantic Similarity**
        doc_texts = [doc.page_content.strip() for doc in retrieved_chunks]
        question_embedding = semantic_model.encode(question, convert_to_tensor=True)
        doc_embeddings = semantic_model.encode(doc_texts, convert_to_tensor=True)

        # **Step 2: Rank the Best Matching Answer**
        scores = util.pytorch_cos_sim(question_embedding, doc_embeddings)[0]
        best_idx = scores.argmax().item()
        best_answer = doc_texts[best_idx]

        # **Step 3: Extract the Most Relevant Sentence**
        sentences = best_answer.split(". ")
        best_sentence = None
        best_score = -1

        for sentence in sentences:
            sentence_embedding = semantic_model.encode(sentence, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(question_embedding, sentence_embedding)[0].item()

            if similarity_score > best_score:
                best_score = similarity_score
                best_sentence = sentence

        # **Step 4: Truncate response if it's too long**
        if best_sentence and len(best_sentence) > 400:
            best_sentence = best_sentence[:400] + "..."

        # **Step 5: Use LLM for Context-Aware Response**
        context = "\n".join([chunk.page_content for chunk in retrieved_chunks])
        refined_response = llm.invoke(f"Answer based on context:\n{context}\n\nQuestion: {question}")

        # Store in cache
        response_cache[cache_key] = refined_response

        return {"response": refined_response or best_sentence}

    except Exception as e:
        logger.exception(f"Error answering question: {str(e)}")
        return {"response": "An error occurred while processing your question. Please try again."}
@app.get("/check_embeddings/")
async def check_embeddings(pdf_name: str):
    """Check if FAISS embeddings exist and log the results to the terminal."""
    db_path = os.path.join(VECTOR_STORE_DIR, pdf_name[:-4])
    
    if os.path.exists(db_path):
        message = f"Embeddings exist for {pdf_name}"
    else:
        message = f"Embeddings NOT found for {pdf_name}"
    
    print(message)  # ✅ Log to the terminal instead of returning JSON
    return {"message": "Check the terminal for embedding status"}

@app.get("/history/")
async def get_chat_history():
    """Retrieve stored chat history."""
    return {"cached_responses": response_cache}

@app.delete("/clear_history/")
async def clear_chat_history():
    """Clears stored chat history."""
    response_cache.clear()
    return {"message": "Chat history cleared."}
@app.get("/check_embeddings/")
async def check_embeddings(pdf_name: str):
    """Load and print FAISS embeddings in the terminal."""
    db_path = os.path.join(VECTOR_STORE_DIR, pdf_name[:-4])

    if not os.path.exists(db_path):
        print(f"❌ No embeddings found for {pdf_name}")
        return {"message": "No embeddings found. Check the terminal."}

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        
        # Retrieve stored embeddings
        total_embeddings = db.index.ntotal  # Number of embeddings stored
        vectors = db.index.reconstruct_n(0, total_embeddings)  # Retrieve all embeddings

        print(f"\n✅ Embeddings found for {pdf_name} (Total: {total_embeddings})")
        for i, vector in enumerate(vectors[:5]):  # Print first 5 embeddings
            print(f"Embedding {i+1}: {vector[:5]}...")  # Print first 5 values of each vector

    except Exception as e:
        print(f"⚠️ Error loading embeddings: {e}")
        return {"message": "Error loading embeddings. Check terminal."}

    return {"message": "Embeddings printed in terminal. Check console output."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
