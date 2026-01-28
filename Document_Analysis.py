import pdfplumber

pdf_path = "/content/google_terms_of_service_en_in.pdf"

output_text_file = "extracted_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    extracted_text = ""
    for page in pdf.pages:
        extracted_text += page.extract_text()

with open(output_text_file, "w") as text_file:
    text_file.write(extracted_text)

print(f"Text extracted and saved to {output_text_file}")

with open("/content/extracted_text.txt", "r") as file:
    document_text = file.read()

# preview the document content
print(document_text[:500])  # preview the first 500 characters

from transformers import pipeline

# load the summarization pipeline
summarizer = pipeline("summarization", model="t5-small")

# summarize the document text (you can summarize parts if the document is too large)
summary = summarizer(document_text[:1000], max_length=150, min_length=30, do_sample=False)
print("Summary:", summary[0]['summary_text'])

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# split text into sentences
sentences = sent_tokenize(document_text)

# combine sentences into passages
passages = []
current_passage = ""
for sentence in sentences:
    if len(current_passage.split()) + len(sentence.split()) < 200:  # adjust the word limit as needed
        current_passage += " " + sentence
    else:
        passages.append(current_passage.strip())
        current_passage = sentence
if current_passage:
    passages.append(current_passage.strip())

    # load the question generation pipeline
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

# function to generate questions using the pipeline
def generate_questions_pipeline(passage, min_questions=3):
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')
    
    # ensure we have at least 3 questions
    questions = [q.strip() for q in questions if q.strip()]
    
    # if fewer than 3 questions, try to regenerate from smaller parts of the passage
    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            additional_input = ' '.join(passage_sentences[i:i+2])
            additional_results = qg_pipeline(f"generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('<sep>')
            questions.extend([q.strip() for q in additional_questions if q.strip()])
    
    return questions[:min_questions]  # return only the top 3 questions

# generate questions from passages
for idx, passage in enumerate(passages):
    questions = generate_questions_pipeline(passage)
    print(f"Passage {idx+1}:\n{passage}\n")
    print("Generated Questions:")
    for q in questions:
        print(f"- {q}")
    print(f"\n{'-'*50}\n")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# function to track and answer only unique questions
def answer_unique_questions(passages, qa_pipeline):
    answered_questions = set()  # to store unique questions

    for idx, passage in enumerate(passages):
        questions = generate_questions_pipeline(passage)

        for question in questions:
            if question not in answered_questions:  # check if the question has already been answered
                answer = qa_pipeline({'question': question, 'context': passage})
                print(f"Q: {question}")
                print(f"A: {answer['answer']}\n")
                answered_questions.add(question)  # add the question to the set to avoid repetition
        print(f"{'='*50}\n")
              
answer_unique_questions(passages, qa_pipeline)


