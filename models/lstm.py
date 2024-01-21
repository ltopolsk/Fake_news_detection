from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

class Custom_LSTM:
    
    def __init__(self, **kwargs) -> None:
        self.model_params = kwargs.get('model_config')
        self.model = self._get_LSTM()
        self.train_params = kwargs.get('train_config')

    def _get_LSTM(self):
        model = Sequential()
        model.add(Input((1,1500)))
        model.add(LSTM(1500))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(**self.model_params)
        return model
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def train(self, data, labels):
        self.model.fit(data, labels, **self.train_params)
    
    def eval(self, data, labels):
        labels_pred = self.model.predict(data)
        labels_pred = labels_pred.astype('int32')
        return {
            'accuracy':accuracy_score(labels_pred, labels),
            'precision':precision_score(labels, labels_pred),
            'recall':recall_score(labels, labels_pred,),
            'f1_score':f1_score(labels, labels_pred,),
            'confusion_matrix':confusion_matrix(labels, labels_pred,)
        }
        