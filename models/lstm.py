from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input

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
        return self.model.evaluate(data, labels, return_dict=True)
