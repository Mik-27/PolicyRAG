import os
import re
from web_scraper import WebScraper
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

import torch
import torch.nn.functional as F
import PyPDF2
import ollama
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils.utils import trim_file_name, verifyPdf
from vdb import VectorDatabase

load_dotenv()
nltk.download('stopwords')
nltk.download('wordnet')


class PolicyRAG():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model.eval()
        self.elastic = VectorDatabase("local")
        ollama.host = os.environ.get("OLLAMA_HOST")
        self.total_token_length = 0
        self.record_count = 0
        

    def pdf_to_text(self, pdf:str) -> str:
        """
            Convert PDF to text using PyPDF2
            pdf: pdf file name to convert
            
            returns: text extracted from the pdf
        """
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
        
    def preprocess_text(self, text:str) -> str:
        """
            Preprocess the text by removing unwanted characters, stopwords, and lemmatizing the words
            text: text to preprocess
            
            returns: preprocessed text
        """
        preprocessed_text = []
        for i, t in enumerate(text):
            t = re.sub(r'\n*\d+\n*$', '', t, flags=re.MULTILINE)
            # t = re.sub(r'==Start of OCR.*?==End of OCR==', '', t, flags=re.DOTALL)

            # Remove Effective and Revision Dates lines:
            t = re.sub(r'Effective:.*?\n', '', t)
            t = re.sub(r'Revised:.*?\n', '', t)

            # General Text Cleaning
            t = re.sub(r"[^a-zA-Z0-9\s\"\'\-\+\=\*\:\;\/\?\(\)\{\}\[\]\!\&\,\.]", '', t)
            t = re.sub(r"\s+", ' ', t).strip()
            t = t.lower()

            # Remove Stopwords
            stop_words = set(stopwords.words('english'))
            words = t.split()
            words = [word for word in words if word not in stop_words]

            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            preprocessed_text.append(" ".join(words)[2:])

        return preprocessed_text
        
        
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
            Pool the last hidden states of the model
            last_hidden_states: last hidden states of the model
            attention_mask: attention mask of the model
            
            returns: pooled last hidden states
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def generate_embeddings(self, text:str) -> Tensor:
        """
            Generate embeddings for the given text using the BGE model
            text: text to generate embeddings for
            
            returns: embeddings and shape of the embeddings
        """
        doc_batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        # Metadata for token counts
        attention_mask = doc_batch_dict['attention_mask']
        token_counts = attention_mask.sum(dim=1).tolist()
        total_tokens = attention_mask.sum().item()
        self.record_count += len(token_counts)
        self.total_token_length += total_tokens
        
        # print(f"Token counts per sequence: {token_counts}")
        # print(f"Total tokens: {total_tokens}")
        
        with torch.no_grad():
            doc_outputs = self.model(**doc_batch_dict)
            doc_embeddings = self.last_token_pool(doc_outputs.last_hidden_state, doc_batch_dict['attention_mask'])

        # Example Input
        # tensor([[-0.0162,  0.2018,  0.0594,  ..., -0.4134, -0.6311,  0.0016],
        # [-0.5256, -1.1052,  0.6363,  ..., -0.0890, -0.2193, -0.4238]]) torch.Size([2, 1024])
        return doc_embeddings.squeeze().cpu().numpy(), doc_embeddings.shape

    def upload_doc(self, doc:str):
        """
            Upload a document to ElasticSearch
            doc: document name to upload
            
            returns: None
        """
        try:
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
            path: path to the directory containing the documents
            
            returns: None
        """
        try:
            pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
            # print("PDF Files:", pdf_files)
            for pdf in pdf_files:
                pdf_path = "./documents/" + pdf
                text = self.pdf_to_text(pdf)
                preprocessed_text = self.preprocess_text(text)
                emb, shape = self.generate_embeddings(text=preprocessed_text)

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
                        self.elastic.push_document(id=int(str(pdf)+str(i)), pdf_path=pdf_path, text=preprocessed_text[i], embedding=e)
                        print("Uploaded - " + pdf + ".pdf")
                    except:
                        raise Exception("Error uploading document - "+pdf+".pdf")
        except:
            raise Exception("Error uploading documents")
        finally:
            print("Total records:", self.record_count)
            print("Total tokens:", self.total_token_length)
            print("Average tokens per record:", self.total_token_length/self.record_count)
        

    def search_docs(self, by:str, query:str):
        """
            Search documents in ElasticSearch DB based on the by parameter
            by: text, embedding, hybrid
            query: query string to search in the documents
            
            Returns: list of documents with their scores
        """
        query_batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            query_outputs = self.model(**query_batch_dict)
            query_emb = self.last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])

        query_emb = query_emb.squeeze().cpu().numpy()
        query_emb = query_emb.tolist()

        # Search query in ElasticSearch DB based in the by parameter
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
        """
            Generate output for the given query and context using the Ollama model
            query: query string to search in the documents
            context: context string to search in the documents
            
            returns: response from the Ollama model
        """
        # Message prompt for the Ollama model
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
    # text = rag.pdf_to_text("1540244.pdf")
    # preprocessed_text = rag.preprocess_text(text)
    # print("Preprocessed Text:", preprocessed_text)
    rag.elastic.create_index(index_name="policy", dims=1024)
    rag.upload_docs(path="./documents/")
    # res = rag.search_docs(by="text", query="Capital Management Group")
    # print(res)
    # test_ollama_connection()