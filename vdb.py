from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from pprint import pprint

import os
import elasticsearch
import numpy as np

load_dotenv()

class VectorDatabase:
    def __init__(self, conn: str) -> None:
        self.index = "policy"
        self.dims = 1024
        try:
            if conn == 'local':
                self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
                if self.es.ping():
                    print("Connected to Elasticsearch")
                    pprint(self.es.info())
                else:
                    print("Connection failed")
            elif conn == 'cloud':
                self.es = Elasticsearch(cloud_id=os.environ['ELASTIC_CLOUD_ID'], api_key=os.environ['ELASTIC_API_KEY'])
                if self.es.ping():
                    print("Connected to Elasticsearch")
                    pprint(self.es.info())
                else:
                    print("Connection failed")
            else:
                raise AttributeError(conn, "argument invalid.")
        except:
            raise elasticsearch.ConnectionError("Cannot establish connection to DB.")
        
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
        results = [{"pdf_path": hit["_source"]["pdf_path"], "text": hit["_source"]["text"]} for hit in hits]

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
        # print(response)
        hits = response['hits']['hits']
        # print(hits)
        results = [{"id":hit["_source"]["id"], "pdf_path": hit["_source"]["pdf_path"], "text": hit["_source"]["text"], "score": hit["_score"]} for hit in hits]
        # print(results)
        return results
    
    
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
        results = [{
            "pdf_path": hit["_source"]["pdf_path"],
            "text": hit["_source"]["text"],
            "score": hit["_score"]
        } for hit in hits]
        
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

    def close(self):
        """
            Close connection to elasticsearch server/localhost
        """
        self.es.close()


if __name__ == "__main__":
    elastic = VectorDatabase(conn="cloud")