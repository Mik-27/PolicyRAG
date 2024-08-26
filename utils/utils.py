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
    try:
        if os.path.isfile("./documents/"+pdf+".pdf"):
            return True
        else:
            raise FileNotFoundError("File does not exist.")
    except:
        raise FileNotFoundError("File does not exist.")