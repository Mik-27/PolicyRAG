from web_scraper import WebScraper
from transformers import AutoTokenizer, AutoModelForCausalLM

import PyPDF2
import os

from utils.utils import verifyPdf

class PolicyRAG():
    def __init__(self):
        pass

    def pdfToText(self, pdf):
        pdf = "1549280"
        text = []
        pdf_path = "./documents/" + pdf + ".pdf"
        if verifyPdf:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    text.append(page_text)

            print(text)
            return text
        else:
            return -1
    
    def generateEmbeddings(self, text):
        tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)

        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        op = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        print(op.shape)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    

if __name__ == "__main__":
    rag = PolicyRAG()
    text = rag.pdfToText("")
    rag.generateEmbeddings(text)

