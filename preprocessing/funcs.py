import nltk
import string
import numpy as np
from gensim.models import Word2Vec
from typing import Iterable

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

def preprocess_text(text:str) -> list[str]:
    """
        Function for preprocessing text. Included steps:
        - punctuation removal
        - lowering text
        - tokenization
        - lemmatization
    """
    ret_text = text.translate(text.maketrans({x:'' for x in string.punctuation})).lower()
    ret_text = nltk.tokenize.word_tokenize(ret_text)
    ret_text = [word for word in ret_text if word not in stopwords]
    ret_text = [lemmatizer.lemmatize(word) for word in ret_text]
    return  ret_text

def get_word2vec(words: Iterable[list[str]], **kwargs) -> Word2Vec:
    """
        Word2Vec model getter. Arguments:
        - words: Iterable[list[str]] -> preprocessed text data
        - other arguments for Word2Vec model (such as vector size, workers, etc.)
    """
    return Word2Vec(words, **kwargs)

def vectorize(words: list[str], model: Word2Vec, first_n_tokens: int, vector_size: int) -> np.ndarray:
    """
        Function for transforming words into vector of numbers. Arguments:
        - words: list[str] -> preprocessed single text data
        - model: Word2Vec
        - first_n_tokens: int -> number of significant tokens.
        - vector_size: int -> size of output vector
        Each row now has first_n_tokens * vector_size elements

    """
    words_vecs = [model.wv[word] for word in words if word in model.wv]
    for _ in range(first_n_tokens-len(words_vecs)):
        words_vecs.append(np.zeros(vector_size,))
    return np.asarray(words_vecs[:first_n_tokens]).flatten()
