from web_scraper import WebScraper
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from elasticsearch import Elasticsearch

import torch
import torch.nn.functional as F
import PyPDF2
import os

from utils.utils import verifyPdf
from vdb import VectorDatabase


class PolicyRAG():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model.eval()

        self.elastic = VectorDatabase("cloud")

    def pdf_to_text(self, pdf:str) -> str:
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
    
    def generate_embeddings(self, text:str) -> Tensor:
        doc_batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            doc_outputs = self.model(**doc_batch_dict)
            doc_embeddings = self.last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])

        # Example Input
        # tensor([[-0.0162,  0.2018,  0.0594,  ..., -0.4134, -0.6311,  0.0016],
        # [-0.5256, -1.1052,  0.6363,  ..., -0.0890, -0.2193, -0.4238]]) torch.Size([2, 1024])
        return doc_embeddings.squeeze().cpu().numpy(), doc_embeddings.shape

    def upload_data(self):
        try:
            pdf = "1549228"
            pdf_path = "./documents/" + pdf + ".pdf"
            text = self.pdf_to_text("")
            emb, shape = self.generate_embeddings(text=text)

            assert shape[1] == self.elastic.dims

            emb = emb.tolist()
            for i, e in enumerate(emb):
                self.elastic.push_document(id=int(str(pdf)+str(i)), pdf_path=pdf_path, text=text, embedding=e)
        except:
            raise Exception("Error uploading document - "+pdf)
        

    def search_docs(self, by:str, query:str):
        query_batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            query_outputs = self.model(**query_batch_dict)
            query_emb = self.last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])

        query_emb = query_emb.squeeze().cpu().numpy()
        query_emb = query_emb.tolist()

        if by == "text":
            results = self.elastic.search_by_text(query, top_k=10)
        elif by == "embedding":
            results = self.elastic.search_by_embedding(query_emb, top_k=10)
        elif by == "hybrid":
            results = self.elastic.hybrid_search(query, query_emb, top_k=10)
        else:
            raise AttributeError("Invalid parameter "+by+" for argument 'by'.")
        
        return results


if __name__ == "__main__":
    rag = PolicyRAG()
    # text = rag.pdf_to_text("")
    # rag.generate_embeddings(text)
    # rag.elastic.create_index(index_name="policy", dims=1024)
    res = rag.search_docs(by="embedding", query="Capital Management Group")
    print(res)

