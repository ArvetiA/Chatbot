import faiss
import numpy as np
import sys
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ Step 1: Get PDF name from the command-line argument
if len(sys.argv) < 2:
    print("❌ Please provide the PDF file name as an argument.")
    print("Usage: python test.py MyUploadedPDF.pdf")
    exit()

pdf_name = sys.argv[1]  # Get PDF name dynamically from command line
db_path = os.path.join("vector_store", pdf_name[:-4])

# ✅ Step 2: Check if embeddings exist
if not os.path.exists(db_path):
    print(f"❌ No embeddings found for {pdf_name}. Please check if embeddings were generated.")
    exit()

# ✅ Step 3: Load embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    total_embeddings = db.index.ntotal  # Get number of stored embeddings
    if total_embeddings == 0:
        print(f"⚠️ Embeddings exist but are empty for {pdf_name}.")
        exit()

    # ✅ Step 4: Print first 5 embeddings
    vectors = [db.index.reconstruct(i) for i in range(min(5, total_embeddings))]
    print(f"\n✅ Found {total_embeddings} embeddings for {pdf_name}")
    for i, vector in enumerate(vectors):
        print(f"Embedding {i+1}: {vector[:5]}...")  # Print first 5 values

    # ✅ Step 5: Generate embedding for a question and search for an answer
    question = input("\nPlease enter your question: ")
    
    # Generate the embedding for the question
    question_embedding = embeddings.embed_query(question)
    print(f"\n✅ Question Embedding: {question_embedding[:5]}...")  # Display first 5 values

    # Perform a similarity search using the question's embedding
    k = 3  # Number of relevant chunks to retrieve
    D, I = db.index.search(np.array([question_embedding]).astype(np.float32), k)

    # Get the corresponding chunks (answers) from the PDF
    print(f"\n✅ Answer (Most Relevant Chunks from PDF):")
    for i in range(k):
        relevant_chunk = db.index.reconstruct(int(I[0][i]))  # Ensure I[0][i] is cast to int
        print(f"Answer {i+1}: {relevant_chunk}")

except Exception as e:
    print(f"⚠️ Error loading FAISS index: {e}")
