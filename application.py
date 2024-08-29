from web_scraper import WebScraper
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from elasticsearch import Elasticsearch

import torch
import torch.nn.functional as F
import PyPDF2
import os

from utils.utils import verifyPdf


class PolicyRAG():
    def __init__(self):
        pass

    def pdfToText(self, pdf:str) -> str:
        pdf = "1549228"
        text = []
        pdf_path = "./documents/" + pdf + ".pdf"
        if verifyPdf:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    text.append(page_text)
            return text
        else:
            return -1
        
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def generateEmbeddings(self, text:str) -> Tensor:
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        model.eval()

        doc_batch_dict = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        print(doc_batch_dict.shape)
        with torch.no_grad():
            doc_outputs = model(**doc_batch_dict)
            doc_embeddings = self.last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])
        
        print(doc_embeddings, doc_embeddings.shape)
        # Example Input
        # tensor([[-0.0162,  0.2018,  0.0594,  ..., -0.4134, -0.6311,  0.0016],
        # [-0.5256, -1.1052,  0.6363,  ..., -0.0890, -0.2193, -0.4238]]) torch.Size([2, 1024])

    def uploadData(self, mode:str):
        pass


if __name__ == "__main__":
    rag = PolicyRAG()
    text = rag.pdfToText("")
    rag.generateEmbeddings(text)

