from dotenv import load_dotenv
from pprint import pprint

import os
from elasticsearch import Elasticsearch, ConnectionError
import numpy as np

load_dotenv()


class VectorDatabase:
    def __init__(self, conn: str) -> None:
        self.index = "policy"
        self.dims = 1024
        try:
            if conn == 'local':
                self.es = Elasticsearch("https://localhost:9200",
                    basic_auth=("elastic", os.environ['ELASTIC_PASSWORD']),
                    verify_certs=False,
                    request_timeout=30)
                if self.es.ping():
                    print("Connected to Elasticsearch")
                    pprint(self.es.info())
                else:
                    raise ConnectionError("Could not establish connection to Elastic DB.")
            elif conn == 'cloud':
                self.es = Elasticsearch(cloud_id=os.environ['ELASTIC_CLOUD_ID'], api_key=os.environ['ELASTIC_API_KEY'])
                if self.es.ping():
                    print("Connected to Elasticsearch")
                    pprint(self.es.info())
                else:
                    raise ConnectionError("Could not establish connection to Elastic DB.")
            elif conn == 'deployment':
                self.es = Elasticsearch(
                    "http://elasticsearch:9200",
                    verify_certs=False,
                    request_timeout=30
                )
                if self.es.ping():
                    print("Connected to Elasticsearch")
                    pprint(self.es.info())
                else:
                    raise ConnectionError("Could not establish connection to Elastic DB.")
            else:
                raise AttributeError(conn, "Argument invalid.")
        except:
            raise ConnectionError("Could not establish connection to Elastic DB.")

       
    def create_index(self, index_name: str, dims: int) -> None:
        try:
            self.index = index_name
            self.dims = dims
            if not self.es.indices.exists(index=index_name):
                res = self.es.indices.create(index=index_name, body={
                    "mappings": {
                        "properties": {
                            "pdf_path": {"type": "keyword"},
                            "text": {"type": "text"},
                            "embedding": {"type": "dense_vector", "dims": dims} 
                        }
                    }
                })
                print(res, "\n Index '"+index_name+"' created successfully.")
            else:
                print("Index '"+index_name+"' already exists.")
        except:
            raise Exception("Error creating Index '"+index_name+"'")
        

    def push_document(self, id:int, pdf_path: str, text: str, embedding: list) -> None:
        if embedding is None or not isinstance(embedding, list):
            raise ValueError(f"Embedding must be a list, got {type(embedding)}")
        
        if len(embedding) != self.dims:
            raise ValueError(f"Embedding length ({len(embedding)}) doesn't match expected dimensions ({self.dims})")

        for i, value in enumerate(embedding):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric value found at position {i}: {value}")
        
        document = {
            'id':id,
            'pdf_path': pdf_path,
            'text': text,
            'embedding': embedding
        }
        res = self.es.index(index=self.index, body=document)
        pprint(res)


    def search_by_text(self, query_text, top_k=5):
        search_query = {
            "query": {
                "match": {
                    "text": query_text
                }
            }
        }

        response = self.es.search(index=self.index, body=search_query)
        hits = response['hits']['hits']
        results = [{"id":hit["_source"]["id"], "pdf_path": hit["_source"]["pdf_path"], "text": hit["_source"]["text"], "score": hit["_score"]} for hit in hits]

        return results


    def search_by_embedding(self, query_embedding, top_k=5):
        # query_embedding = np.array(query_embedding).reshape(1, -1)
        print(len(query_embedding))
        search_query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}  # Retrieve all documents
                    },
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

        response = self.es.search(index=self.index, body=search_query)
        hits = response['hits']['hits']
        results = [{"id":hit["_source"]["id"], "pdf_path": hit["_source"]["pdf_path"], "text": hit["_source"]["text"], "score": hit["_score"]} for hit in hits]
        
        return results
    
    # Not working
    def hybrid_search(self, query, query_embedding, top_k=5):
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": query
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        
        response = self.es.search(index=self.index, body=search_query)
        hits = response['hits']['hits']
        results = [{"id":hit["_source"]["id"], "pdf_path": hit["_source"]["pdf_path"], "text": hit["_source"]["text"], "score": hit["_score"]} for hit in hits]
        
        return results


    # def get_relevant_docs(self, query_embedding: list):
    #     # To be updated
    #     query_string = {
    #         "field": "title_embedding",
    #         "query_vector": query_embedding,
    #         "k": 1,
    #         "num_candidates": 100
    #     }
    #     results = self.es.search(index=self.index, knn=query_string, source_includes=["title", "genre", "release_year"])
    #     print(results['hits']['hits'])


    def update_doc(self, id:int):
        # Make updates
        update_doc = {
            'doc': {
                'content': 'Updated content for this document.'
            }
        }
        self.es.update(index=self.index, id=id, body=update_doc)
        
    def count_documents(self):
        """Count documents in the index"""
        try:
            result = self.es.count(index=self.index)
            count = result["count"]
            print(f"Document count in {self.index}: {count}")
            return count
        except Exception as e:
            print(f"Error counting documents: {e}")
            return -1
        
    def delete_all(self):
        """
            Delete all documents in the index
        """
        self.es.delete_by_query(index=self.index, body={"query": {"match_all": {}}}, refresh=True)
        print("All documents deleted from index.")

    def close(self):
        """
            Close connection to elasticsearch server/localhost
        """
        self.es.close()


if __name__ == "__main__":
    elastic = VectorDatabase(conn="local")
    elastic.count_documents()
    # es = Elasticsearch("https://localhost:9200",
    #         basic_auth=("elastic", os.environ['ELASTIC_PASSWORD']),
    #         verify_certs=False,
    #         request_timeout=90)
    # print(es.ping())