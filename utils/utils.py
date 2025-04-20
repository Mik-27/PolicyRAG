import requests
import os

def checkUrlHealth(url):
    """
        Check whether provided URL is valid or not.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return 1
        else:
            response.raise_for_status()
            return 0
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    
def verifyPdf(pdf):
    """
        Verify whether the provided PDF file exists or not.
    """
    try:
        if os.path.isfile("./documents/"+pdf+".pdf"):
            return True
        else:
            raise FileNotFoundError("File does not exist.")
    except:
        raise FileNotFoundError("File does not exist.")
  
    
def trim_file_name(file_name):
    """
        Trim the file name to remove unwanted characters.
    """
    # Remove unwanted characters from the file name
    trimmed_file_name = file_name.split(".pdf")[0]
    
    # Ensure the file name is not empty
    if not trimmed_file_name:
        raise ValueError("File name cannot be empty after trimming.")
    
    return trimmed_file_name