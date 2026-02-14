from collections import defaultdict,Counter
from lib.ulits import PROJECT_ROOT, load_data
import os
import math
import pickle

CACHE_PATH = PROJECT_ROOT / 'cache'

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.term_frequencies=defaultdict(Counter) # document_id : counter
        self.docmap = {}
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.indexpath = CACHE_PATH / 'index.pkl'
        self.term_frequencies_path= CACHE_PATH / 'term_frequencies.pkl'

    def __add_document(self, doc_id, text):
        from lib.keywork_search import tokenize_text
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
            

    def get_documents(self, term):
        return sorted(list(self.index.get(term, [])))

    def build(self):
        movies = load_data()
        for movie in movies:
            doc_id = movie["id"]
            text = f'{movie["title"]} {movie["description"]}'
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.indexpath, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
    def load(self):
        try:
            with open(self.indexpath, 'rb') as f:
                self.index = pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")

        try:
            with open(self.docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
        except Exception as e:
            print(f"Error loading docmap: {e}")
        try:
            with open(self.term_frequencies_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)
        except Exception as e:
            print(f"Error loading docmap: {e}")

    def get_tf(self, doc_id, term):
        from lib.keywork_search import tokenize_text
        term=tokenize_text(term)
        if len(term)>1:
            raise Exception("More than one token is passed")
        term_token = term[0]
        count = self.term_frequencies[doc_id].get(term_token, 0)
        if count:
            return count
        else:
            return 0

    def get_idf(self,term):
        from lib.keywork_search import tokenize_text
        term=tokenize_text(term)
        if len(term)>1:
            raise Exception("More than one token is passed")
        term_token = term[0]
        total_doc_count=len(self.docmap)
        term_match_doc_count=len(self.index[term_token])
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tfidf(self,term,doc_id):
        from lib.keywork_search import tokenize_text
        term=tokenize_text(term)
        if len(term)>1:
            raise Exception("More than one token is passed")
        term_token = term[0]
        total_doc_count=len(self.docmap)
        term_match_doc_count=len(self.index[term_token])
        idf=math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        tf=self.term_frequencies[doc_id].get(term_token, 0)
        
        return idf*tf
    
def tfidf_command(term,doc_id):
    obj = InvertedIndex()
    obj.load()
    tf_idf=obj.get_tfidf(term,doc_id)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def build_command():
    obj = InvertedIndex()
    obj.build()
    obj.save()

def tf_command(doc_id, term):
    obj = InvertedIndex()
    obj.load()
    tf = obj.get_tf(doc_id, term)
    return tf

def idf_command(term):
    obj = InvertedIndex()
    obj.load()
    idf=obj.get_idf(term=term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")