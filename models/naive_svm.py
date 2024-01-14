import pickle
from sklearn import naive_bayes, svm

class BaseModel:

    def __init__(self, **kwargs) -> None:
        self.model = None

    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))

    def train(self, data, labels, **kwargs):
        self.model.fit(data, labels)
    
    def predict(self, data):
        return self.model.predict(data)

class NaiveBayes(BaseModel):

    def __init__(self, **kwargs) -> None:
        self.model = naive_bayes.GaussianNB(**kwargs)

class SVM(BaseModel):

    def __init__(self, **kwargs) -> None:
        self.model = svm.SVC(**kwargs)
