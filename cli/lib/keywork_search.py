from lib.ulits import load_stop_words
import string
from nltk.stem import PorterStemmer
from lib.InvertedIndex import InvertedIndex


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text):
    stemmer = PorterStemmer()
    stop_words = load_stop_words()
    text = clean_text(text)

    tokens = [stemmer.stem(tk) for tk in text.split() if tk and tk not in stop_words]
    return tokens


def match_tokens(query_tokens, movie_tokens):
    movie_set = set(movie_tokens)
    return any(q in movie_set for q in query_tokens)


def search_command(query, n_result=5):
    obj = InvertedIndex()
    obj.load()

    seen = set()
    results = []
    query_tokens = tokenize_text(query)

    for qt in query_tokens:
        matching_docs_ids = obj.get_documents(qt)

        for doc_id in matching_docs_ids:
            if doc_id in seen:
                continue

            seen.add(doc_id)
            results.append(obj.docmap[doc_id])

            if len(results) >= n_result:
                break

        if len(results) >= n_result:
            break

    return results
