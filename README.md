# Chat with PDF

A simple chat application that allows you to upload a PDF document and ask questions about its content. Built with Chainlit, Google Gemini API, LlamaParse, and ChromaDB.

## Features

- 📄 Upload PDF documents
- 💬 Ask questions about the document content
- 🤖 Powered by Google Gemini AI
- 📝 Advanced PDF parsing with LlamaParse
- 🔍 Vector search using ChromaDB for accurate answers

## Prerequisites

- Python 3.9 or higher
- Google API Key (for Gemini)
- LlamaCloud API Key (for LlamaParse)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/apkarthik1986/ChatWithPDF.git
   cd ChatWithPDF
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API Keys:**
   - Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Get your LlamaCloud API key from [LlamaCloud](https://cloud.llamaindex.ai/api-key)
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your API keys

## Running the Application

Start the Chainlit application:

```bash
chainlit run app.py
```

The application will open in your browser at `http://localhost:8000`.

## Usage

1. When the app starts, you'll be prompted to upload a PDF file
2. After uploading, the PDF will be processed and indexed
3. Ask any questions about the document content
4. The AI will provide answers based on the document

## Project Structure

```
ChatWithPDF/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
├── .env                # Your environment variables (not committed)
├── chroma_db/          # Vector database storage (auto-created)
└── README.md           # This file
```

## Technologies Used

- **[Chainlit](https://chainlit.io/)** - Chat interface framework
- **[LangChain](https://langchain.com/)** - LLM application framework
- **[Google Gemini](https://ai.google.dev/)** - AI model for chat and embeddings
- **[LlamaParse](https://cloud.llamaindex.ai/)** - Advanced PDF parsing
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for document search

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.