import os
from dotenv import load_dotenv
import ollama
from transformers import AutoTokenizer, AutoModel
from src.chunking import RecursiveChunker
from src.vdb import VectorDatabase
from utils import trim_file_name
from .io import pdf_to_text
from .preprocess import preprocess_text
from .embeddings import generate_embeddings

import torch

load_dotenv()


class PolicyRAG:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True)
        self.model.eval()
        self.elastic = VectorDatabase("deployment")
        self.chunker = RecursiveChunker(max_tokens=512, min_tokens=100, overlap_tokens=50)
        ollama.host = os.environ.get("OLLAMA_HOST")
        self.total_token_length = 0
        self.record_count = 0

    def chunk_text(self, text_list: list) -> list:
        return self.chunker.chunk_document(text_list)

    def pdf_to_text(self, pdf: str):
        return pdf_to_text(pdf)

    def preprocess_text(self, text_input):
        return preprocess_text(text_input)

    def generate_embeddings(self, text):
        return generate_embeddings(self.tokenizer, self.model, text, stats_obj=self)

    def upload_docs(self, path: str, use_recursive_chunking: bool = True):
        # keep logic moved from application.py but call local helpers
        try:
            pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
            for pdf in pdf_files:
                pdf_path = "./documents/" + pdf
                text = self.pdf_to_text(pdf)

                if use_recursive_chunking:
                    chunks = self.chunk_text(text)
                    chunks = self.preprocess_text(chunks)
                    processed_contents = [chunk['processed_content'] for chunk in chunks]
                    emb, shape = self.generate_embeddings(text=processed_contents)

                    assert shape[1] == self.elastic.dims
                    emb = emb.tolist()
                    if shape[0] == 1:
                        emb = [emb]

                    pdf_trimmed = trim_file_name(pdf)
                    if not pdf_trimmed:
                        raise Exception("Error trimming file name")

                    for i, (chunk, embedding) in enumerate(zip(chunks, emb)):
                        chunk_id = int(str(pdf_trimmed) + str(chunk['chunk_index']))
                        self.elastic.push_document(
                            id=chunk_id,
                            pdf_path=pdf_path,
                            text=chunk['processed_content'],
                            embedding=embedding,
                            original_text=chunk['content'],
                            page_number=chunk['page'],
                            section_header=chunk['section'],
                            chunk_index=chunk['chunk_index']
                        )
                    print(f"Uploaded {len(chunks)} chunks from {pdf_trimmed}.pdf")
                else:
                    preprocessed_text = self.preprocess_text(text)
                    emb, shape = self.generate_embeddings(text=preprocessed_text)
                    assert shape[1] == self.elastic.dims
                    emb = emb.tolist()
                    if shape[0] == 1:
                        emb = [emb]
                    for i, e in enumerate(emb):
                        pdf_trimmed = trim_file_name(pdf)
                        if not pdf_trimmed:
                            raise Exception("Error trimming file name")
                        self.elastic.push_document(
                            id=int(str(pdf_trimmed)+str(i)),
                            pdf_path=pdf_path,
                            text=preprocessed_text[i],
                            embedding=e
                        )
        except Exception:
            raise
        finally:
            print("Total records:", self.record_count)
            print("Total tokens:", self.total_token_length)
            if self.record_count > 0:
                print("Average tokens per record:", self.total_token_length/self.record_count)

    def search_docs(self, by: str, query: str):
        query_batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            query_outputs = self.model(**query_batch_dict)
            query_emb = self.last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])

        query_emb = query_emb.squeeze().cpu().numpy().tolist()

        if by == "text":
            results = self.elastic.search_by_text(query, top_k=10)
        elif by == "embedding":
            results = self.elastic.search_by_embedding(query_emb, top_k=10)
        elif by == "hybrid":
            results = self.elastic.hybrid_search(query, query_emb, top_k=10)
        else:
            raise AttributeError("Invalid parameter "+by+" for argument 'by'.")
        return results

    # reuse last_token_pool from embeddings module if needed by external callers

    def generate_query_output(self, query: str, context: str):
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Answer the following question on ASU Policies:" 
            + query
            + "by using the following text:"
            + context},
        ]
        outputs = ollama.chat(model='gemma2:2b', messages=messages)
        return outputs['message']['content']
