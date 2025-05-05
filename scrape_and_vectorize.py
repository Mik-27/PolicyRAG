import os
import logging
import time
from dotenv import load_dotenv
from web_scraper import WebScraper
from application import PolicyRAG
from vdb import VectorDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vectorize-docs')

load_dotenv()

def scrape_and_vectorize():
    """
    Main function to scrape documents and upload them to the vector database
    """
    start_time = time.time()
    
    try:
        # Initialize vector database and create index
        logger.info("Initializing vector database connection")
        elastic = VectorDatabase("deployment")
        
        # Create the index with 1024 dimensions (from your embeddings model)
        logger.info("Creating/verifying index")
        elastic.create_index(index_name="policy", dims=1024)
        
        # Initialize document scraper if URL is provided
        scrape_url = "https://public.powerdms.com/ASU/tree"
        if scrape_url:
            logger.info(f"Scraping documents from {scrape_url}")
            scraper = WebScraper(scrape_url)
            scraper.getDocuemnts()
            logger.info("Document scraping completed")
        else:
            logger.info("No SCRAPE_URL provided - skipping scraping")
        
        # Initialize PolicyRAG and upload documents
        logger.info("Initializing embedding model and starting document processing")
        rag = PolicyRAG()
        rag.upload_docs(path="./documents/")
        
        # Report statistics
        doc_count = elastic.count_documents()
        logger.info(f"Documents successfully vectorized and uploaded: {doc_count}")
        
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in scrape_and_vectorize: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting document scraping and vectorization process")
    success = scrape_and_vectorize()
    if success:
        logger.info("Process completed successfully")
        exit(0)
    else:
        logger.error("Process failed")
        exit(1)