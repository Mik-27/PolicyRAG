import PyPDF2
from typing import List
from utils import verifyPdf


def pdf_to_text(pdf: str, documents_dir: str = "./documents/") -> List[str]:
    """
    Convert a PDF file to a list of page texts.
    """
    text = []
    pdf_path = documents_dir.rstrip('/') + '/' + pdf
    if verifyPdf:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                text.append(page_text)
        return text
    else:
        return -1
