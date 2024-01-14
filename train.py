import json
import argparse
import numpy as np
from preprocessing.preprocess import preprocess
from models.naive_svm import NaiveBayes, SVM
from models.lstm import Custom_LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def get_args():
    args = argparse.ArgumentParser(description='Script for training and saving models')
    args.add_argument('--config_file', default='config.json', action='store')
    args.add_argument('--model', action='store', choices=('bayes', 'svm', 'lstm'))
    args.add_argument('--filename', action='store', default='model')
    return args.parse_args()

def train_and_save(model, filename, data, labels, **kwargs):

    model.train(data, labels, **kwargs)
    model.save(config['models_path'] + '/'+filename)

if __name__=="__main__":
    
    args = get_args()
    with open(args.config_file) as f:
        config = json.load(f)


    merged_data = preprocess(**config)

    transformed_array_X = np.asarray(merged_data['text'].tolist())
    X_train, X_test, Y_train, Y_test = train_test_split(transformed_array_X, merged_data['label'], **config['split_data_args'])
    models = {
        'bayes': NaiveBayes(),
        'svm': SVM(**config['SVM_config']),
        'lstm': Custom_LSTM(**config['LSTM_config'])
    }
    if args.model == 'lstm':
        X_train2 = []
        for x in X_train:
            X_train2.append([x])
        X_train = np.asarray(X_train2)
        X_test2 = []
        for x in X_test:
            X_test2.append([x])
        X_test = np.asarray(X_test2)
    model = models[args.model]
    train_and_save(model, args.filename, X_train, Y_train)
    print(model.eval(X_test, Y_test))