📄 PDF-Insight: Automated Q&A & Summarization Pipeline
A streamlined RAG (Retrieval-Augmented Generation) application that converts static PDF documents into interactive knowledge bases. This tool extracts text, summarizes content, automatically generates relevant questions, and provides precise answers using state-of-the-art Transformer models.

🚀 Overview
Reading long legal documents or technical manuals is time-consuming. PDF-Insight automates the "study" process by:
Extracting raw text from PDF files.
Chunking text into manageable passages to preserve context.
Generating intelligent questions based on the content.
Answering those questions using a pre-trained deep learning model.

✨ Features
Smart Extraction: Uses pdfplumber to handle complex PDF layouts better than standard libraries.
Automatic Question Generation: Employs a T5-Base model to "act like a teacher" and create comprehension questions.
High-Confidence QA: Uses RoBERTa-Base (SQuAD 2.0) to find exact answers within the text.
Interactive Web UI: A built-in Gradio interface allows users to upload files and view results in a clean, tabular format.
Duplicate Filtering: Logic to ensure only unique, high-confidence Q&A pairs are presented.

🛠️ Tech Stack
Models: Hugging Face Transformers (T5, RoBERTa)
UI: Gradio
Text Processing: NLTK (Tokenization), pdfplumber
Data Handling: Pandas

📥 Installation
Clone the repository:

Bash
git clone https://github.com/your-username/pdf-insight.git
cd pdf-insight
Install dependencies:

Bash
pip install transformers torch pdfplumber nltk gradio pandas
Download NLTK data:

Python
import nltk
nltk.download('punkt')
🖥️ Usage
Run the main script to launch the Gradio web interface:

Bash
python app.py
Once running, open the local URL (e.g., http://127.0.0.1:7860) in your browser, upload a PDF, and click Process Document.

🧠 How It Works (The Pipeline)
Preprocessing: PDF → Text → Sentences.

Chunking: Sentences are grouped into ~200-word "Passages" to fit within model token limits.

QG Phase: The Question Generation model scans each passage for key facts.

QA Phase: The Question Answering model cross-references the generated questions against the passage to find the answer.

Scoring: Only answers with a confidence score above a certain threshold are displayed to minimize "hallucinations."
