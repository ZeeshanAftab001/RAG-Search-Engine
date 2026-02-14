import json
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
print(__file__)

MOVIES_PATH=PROJECT_ROOT/"data"/'movies.json'
STOP_WORDS_PATH=PROJECT_ROOT/"data"/'stopwords.txt'


def load_data() -> list[dict]:

    with open(MOVIES_PATH,'r') as f:
        data=json.load(f)
    return data["movies"]
    

def load_stop_words():
    with open(STOP_WORDS_PATH,'r') as f:
        stop_words=f.read()
    stop_words=stop_words.splitlines()
    return stop_words