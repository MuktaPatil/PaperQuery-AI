# PDF Chatbot – Ask Questions to Your Research Paper!
Have a technical research paper and no time to read it all? This chatbot reads your PDF, splits it into chunks, and uses machine learning to answer your questions intelligently.

This project combines PyMuPDF for PDF parsing and HuggingFace Transformers (DistilBERT) for extractive question answering. Perfect for summarizing or querying academic papers, especially in the biomedical or signal processing domain.

## 📌 Features
✅ Extracts text from a PDF (via fitz, aka PyMuPDF)

🔍 Splits content into manageable chunks

💡 Uses a QA model to answer user questions based on content

🤖 Includes a chat-like loop with pretty printed user and bot interaction

📄 Built-in support for GPT2 for future expansion (currently unused)

## 🖥️ Sample Interaction
vbnet
Copy code
🔍 Loading and processing PDF...
✅ Loaded 36 chunks.

✨ Chatbot is ready to answer your questions! Type 'exit', 'quit', or 'stop' to end.

🧠 You: What is the paper about?
Chatbot 🤖: The prediction module of the project also helps identify the probability of seizure onset

🧠 You: what does the paper do?
Chatbot 🤖: exploring innovative approaches that leverage both signal processing and machine learning

🧠 You: what is a seizure?
Chatbot 🤖: correlation coefficients in the time domain and their corresponding eigenvalues

🧠 You: does it have maths?
Chatbot 🤖: Machine learning models have demonstrated the ability to learn complex patterns from diverse datasets

Chatbot 🤖: Goodbye! 👋

### Install required libraries before running:
pip install transformers faiss-cpu torch fitz pymupdf
Note: fitz refers to pymupdf, but some environments may require both.

## 🚀 Usage
Clone the repo and upload your PDF to the workspace.

Replace the pdf_path with the name of your PDF file.

Run the notebook and start chatting with your paper!

## 🧠 Under the Hood
Text Extraction: fitz reads all pages and joins their text.

Chunking: The full text is divided into 500-character chunks to avoid token overflow.

Question Answering: HuggingFace's pipeline("question-answering") model scans chunks for context.

Best Match Logic: The answer with the longest response (proxy for confidence) is selected.

## 📚 Example PDF Used
📄 Seizure Detection and Probability Prediction using Random Forests

This paper explores combining EEG signal features and machine learning (random forests) to detect seizure onset with probability estimates.

### 🙌 Author
Built with 💻, 🧠, and way too much ☕ by Mukta Patil

