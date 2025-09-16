import requests


def checkUrlHealth(url: str) -> int:
    """
    Check whether provided URL is reachable. Returns 1 for OK, raises on error.
    Kept signature and semantics from previous implementation for compatibility.
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
