# Fake_news_detection

## Project setup

1. Clone repository
2. Download and extract data from https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data
3. [recommended] Create python virtual enviroment
``` bash
    python3 -m venv .venv && . .venv/bin/activate
```
4. Download requirements
``` bash
    pip install -r requirements.txt 
```

## Chapter
For now there is some text preprocessing (research work) in pipeline.ipynb. Steps:
- Punctuation removal
- Text lowercasing
- Tokenization
- Lemmatization
- Words embedding using Word2Vec
