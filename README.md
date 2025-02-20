# Chatbot
# RAG-Based PDF Chatbot

## Introduction
This project is a Retrieval-Augmented Generation (RAG) based chatbot designed to process and answer questions based on uploaded PDF documents. It leverages semantic search with FAISS and a language model (Llama 3.2:1B) to provide contextually relevant responses.

## Key Features
- **File Upload**: Users can upload PDF files, which are processed for text extraction and FAISS embedding generation.
- **Chat Interface**: A clean, responsive chat UI for user interaction.
- **Semantic Search**: Utilizes SentenceTransformer and FAISS for embedding-based text retrieval.
- **Context-Aware Answering**: Uses the Llama 3.2:1B model to generate refined responses.
- **Chat History Management**: View past interactions, clear chat history, and start new sessions.

## System Architecture
### Frontend
- Developed using **HTML, Tailwind CSS, and JavaScript**.
- **UI Components**:
  - Sidebar for chat history and session controls.
  - Chatbox for real-time conversation.
  - File upload section.
  - Notifications for system feedback.

### Backend
- **Built with FastAPI**.
- **Technologies Used**:
  - **PyPDF**: Extracts text from PDFs.
  - **FAISS**: Efficient storage and retrieval of embeddings.
  - **SentenceTransformer**: Converts text into vector embeddings.
  - **Llama Model**: Provides AI-powered contextual answers.
- **API Endpoints**:
  - `/upload/`: Handles PDF uploads and embedding generation.
  - `/ask/`: Processes user queries and retrieves relevant answers.
  - `/history/`: Returns stored responses for review.
  - `/clear_history/`: Deletes stored chat history.
  - `/check_embeddings/`: Verifies FAISS embedding status.

## User Workflow
1. **Upload a PDF**: The system extracts text and generates embeddings.
2. **Ask Questions**: The chatbot retrieves relevant text and generates a response.
3. **View Chat History**: Users can review or clear past interactions.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```
5. Open the frontend in a browser.

## Future Enhancements
- Support for additional file formats (Word, Excel, etc.).
- Improved response accuracy using advanced language models.
- Multi-file support for cross-document querying.

## Conclusion
This chatbot integrates a powerful NLP model and semantic search to enable an interactive Q&A system based on user-uploaded PDFs. It serves as a strong foundation for further improvements and scalability.



