import gradio as gr
import pdfplumber
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# --- 1. Load Models & Setup (Run once) ---
print("Loading models... please wait.")
nltk.download('punkt')

# Load pipelines globally so they don't reload on every button click
summarizer = pipeline("summarization", model="t5-small")
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# --- 2. Your Helper Function ---
# (I kept your exact logic for generating at least 3 questions)
def generate_questions_pipeline(passage, min_questions=3):
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')
    
    # Ensure we have at least 3 questions
    questions = [q.strip() for q in questions if q.strip()]
    
    # If fewer than 3 questions, try to regenerate from smaller parts
    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            additional_input = ' '.join(passage_sentences[i:i+2])
            additional_results = qg_pipeline(f"generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('<sep>')
            questions.extend([q.strip() for q in additional_questions if q.strip()])
    
    return questions[:min_questions]

# --- 3. Main Processing Function for Gradio ---
def process_pdf(file_obj):
    # A. Extract Text using pdfplumber
    extracted_text = ""
    # file_obj.name gives the temporary path of the uploaded file
    with pdfplumber.open(file_obj.name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text

    if not extracted_text:
        return "No text found in PDF.", pd.DataFrame()

    # B. Generate Summary
    # Summarizing the first 1000 chars as per your original code
    summary_result = summarizer(extracted_text[:1000], max_length=150, min_length=30, do_sample=False)
    summary_text = summary_result[0]['summary_text']

    # C. Chunking (Your logic)
    sentences = sent_tokenize(extracted_text)
    passages = []
    current_passage = ""
    for sentence in sentences:
        if len(current_passage.split()) + len(sentence.split()) < 200:
            current_passage += " " + sentence
        else:
            passages.append(current_passage.strip())
            current_passage = sentence
    if current_passage:
        passages.append(current_passage.strip())

    # D. Generate Q&A (Your loop logic, adapted for output)
    qa_data = []
    answered_questions = set()

    # NOTE: Limit to first 5 passages for speed in the web demo. 
    # Remove '[:5]' to process the whole document.
    for passage in passages[:5]: 
        questions = generate_questions_pipeline(passage)

        for question in questions:
            if question not in answered_questions:
                # Answer the question
                answer = qa_pipeline({'question': question, 'context': passage})
                
                # Optional: Only show answers with decent confidence scores (> 0.01)
                if answer['score'] > 0.01:
                    qa_data.append([question, answer['answer']])
                    answered_questions.add(question)

    # Convert list to DataFrame for the table display
    df = pd.DataFrame(qa_data, columns=["Generated Question", "Answer"])
    
    return summary_text, df

# --- 4. Launch Gradio Interface ---
demo = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload PDF Document", file_types=[".pdf"]),
    outputs=[
        gr.Textbox(label="Document Summary", lines=3),
        gr.Dataframe(label="Q&A Pairs", headers=["Generated Question", "Answer"], wrap=True)
    ],
    title="PDF Intelligent Q&A Generator",
    description="Upload a PDF. The AI will summarize it and generate unique questions with answers."
)

if __name__ == "__main__":
    demo.launch()
