from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input

class Custom_LSTM:
    
    def __init__(self, **model_args) -> None:
        self.model = self._get_LSTM(**model_args)

    def _get_LSTM(self, **model_args):
        model = Sequential()
        model.add(Input((1,1500)))
        model.add(LSTM(1500))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(**model_args)
        return model
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)

    def train(self, data, labels, **kwargs):
        self.model.fit(data, labels, **kwargs)
