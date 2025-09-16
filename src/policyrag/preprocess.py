import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# Keep behavior as before for now (downloads at import-time)
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text_input) -> list:
    """
    Preprocess text or chunk dicts produced by the chunker.
    Accepts the same input formats as the original method in application.py.
    """
    is_dict_format = isinstance(text_input, list) and len(text_input) > 0 and isinstance(text_input[0], dict)

    if is_dict_format:
        text_list = [chunk['content'] for chunk in text_input]
        chunk_metadata = text_input
    else:
        text_list = text_input
        chunk_metadata = None

    preprocessed_text = []
    for i, t in enumerate(text_list):
        t = re.sub(r'\n*\d+\n*$', '', t, flags=re.MULTILINE)
        t = re.sub(r'Effective:.*?\n', '', t)
        t = re.sub(r'Revised:.*?\n', '', t)
        t = re.sub(r"[^a-zA-Z0-9\s\"\'\-\+\=\*\:\;\/\?\(\)\{\}\[\]\!\&\,\.]", '', t)
        t = re.sub(r"\s+", ' ', t).strip()
        t = t.lower()

        stop_words = set(stopwords.words('english'))
        words = t.split()
        words = [word for word in words if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        # preserve previous behavior that sliced off first two chars
        preprocessed_text.append(" ".join(words)[2:])

    if is_dict_format:
        for i, chunk in enumerate(chunk_metadata):
            chunk['processed_content'] = preprocessed_text[i]
        return chunk_metadata
    else:
        return preprocessed_text
