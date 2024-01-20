import pandas as pd
import numpy as np
from .funcs import preprocess_text, get_word2vec, vectorize, save_word2vec, load_word2vec

def preprocess(**kwargs)->pd.DataFrame:
    """
        Data preprocessing
    """
    fake_data = pd.read_csv(kwargs.get('fake_data_path'))
    fake_data['label'] = 0

    true_data = pd.read_csv(kwargs.get('true_data_path'))
    true_data['label'] = 1
    
    merged_data = pd.concat((fake_data, true_data))
    merged_data['text'] = merged_data['title'] + ' ' + merged_data['text']
    merged_data = merged_data.drop('subject', axis=1).drop('date', axis=1).drop('title', axis=1)
    
    random_permutation = np.random.permutation(len(merged_data))
    merged_data = merged_data.iloc[random_permutation]
    
    if not kwargs.get('all_data', False): 
        merged_data = merged_data.head(kwargs.get('num_rows', 1000))

    merged_data['text'] = merged_data['text'].apply(preprocess_text)
    w2v_text = get_word2vec(merged_data['text'], **kwargs.get('Word2Vec_args'))
    save_word2vec(w2v_text, kwargs.get('word2vec_path'))
    merged_data['text'] = merged_data['text'].apply(lambda x: vectorize(x, w2v_text, kwargs.get('first_n_tokens', 15), kwargs.get('vec_size', 100)))

    return merged_data

def preprocess_test(**kwargs)->pd.DataFrame:
    """
        Test data preprocessing
    """
    
    merged_data = pd.read_csv(kwargs.get('test_data_path'))
    merged_data['text'] = merged_data['title'] + ' ' + merged_data['text']
    merged_data = merged_data.drop('subject', axis=1).drop('date', axis=1).drop('title', axis=1)
    
    if not kwargs.get('all_data', False): 
        merged_data = merged_data.head(kwargs.get('num_rows', 1000))

    merged_data['text'] = merged_data['text'].apply(preprocess_text)
    w2v_text = load_word2vec(kwargs.get('word2vec_path'))
    merged_data['text'] = merged_data['text'].apply(lambda x: vectorize(x, w2v_text, kwargs.get('first_n_tokens', 15), kwargs.get('vec_size', 100)))

    return merged_data
