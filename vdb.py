from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from pprint import pprint

import os
import elasticsearch

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


    def get_relevant_docs(self, query_embedding: list):
        # To be updated
        query_string = {
            "field": "title_embedding",
            "query_vector": query_embedding,
            "k": 1,
            "num_candidates": 100
        }
        results = self.es.search(index=self.index, knn=query_string, source_includes=["title", "genre", "release_year"])
        print(results['hits']['hits'])


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