import os
from web_scraper import WebScraper
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
import PyPDF2
import ollama

from utils.utils import trim_file_name, verifyPdf
from vdb import VectorDatabase

load_dotenv()


class PolicyRAG():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model.eval()
        self.elastic = VectorDatabase("local")
        ollama.host = os.environ.get("OLLAMA_HOST")
        

    def pdf_to_text(self, pdf:str) -> str:
        # pdf = "1549228"
        text = []
        pdf_path = "./documents/" + pdf
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

    def upload_doc(self, doc:str):
        """
            Upload document to ElasticSearch
        """
        try:
            # pdf = "1549228"
            pdf_path = "./documents/" + doc
            text = self.pdf_to_text(doc)
            emb, shape = self.generate_embeddings(text=text)

            assert shape[1] == self.elastic.dims

            emb = emb.tolist()
            for i, e in enumerate(emb):
                self.elastic.push_document(id=int(str(doc)+str(i)), pdf_path=pdf_path, text=text, embedding=e)
        except:
            raise Exception("Error uploading document - "+doc+".pdf")
        
    def upload_docs(self, path:str):
        """
            Upload document to ElasticSearch
        """
        # TODO: Error handling for each documents
        try:
            pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
            # print("PDF Files:", pdf_files)
            for pdf in pdf_files:
                pdf_path = "./documents/" + pdf
                text = self.pdf_to_text(pdf)
                emb, shape = self.generate_embeddings(text=text)

                assert shape[1] == self.elastic.dims

                emb = emb.tolist()
                if shape[0] == 1:
                    emb = [emb]
                # print(shape)

                for i, e in enumerate(emb):
                    pdf = trim_file_name(pdf)
                    if not pdf:
                        raise Exception("Error trimming file name")
                    try:
                        self.elastic.push_document(id=int(str(pdf)+str(i)), pdf_path=pdf_path, text=text[i], embedding=e)
                        print("Uploaded - " + pdf + ".pdf")
                    except:
                        raise Exception("Error uploading document - "+pdf+".pdf")
        except:
            raise Exception("Error uploading documents")
        

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
    
    def generate_query_output(self, query:str, context:str):
        print(type(query), type(context))
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Answer the following question on ASU Policies:" 
            + query 
            + "by using the following text:" 
            + context},
        ]

        outputs = ollama.chat(model='gemma2:2b', messages=messages)

        return outputs['message']['content']


def test_ollama_connection():
    try:
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        import ollama
        ollama.host = ollama_host
        
        # List available models to test connection
        models = ollama.list()
        print(f"Connected to Ollama. Available models: {models}")
        return True
    except Exception as e:
        print(f"{e}")
        return False


if __name__ == "__main__":
    rag = PolicyRAG()
    # rag.elastic.create_index(index_name="policy", dims=1024)
    rag.upload_docs(path="./documents/")
    # res = rag.search_docs(by="text", query="Capital Management Group")
    # print(res)
    # test_ollama_connection()