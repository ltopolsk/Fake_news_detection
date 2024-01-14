import pickle
from sklearn import naive_bayes, svm

class BaseModel:

    def __init__(self, **kwargs) -> None:
        self.model = None

    def save(self, path):
        with open(path, 'w') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'r') as f:
            self.model = pickle.load(f)

    def train(self, data, labels, **kwargs):
        self.model.fit(data, labels)    

class NaiveBayess(BaseModel):

    def __init__(self, **kwargs) -> None:
        self.model = naive_bayes.GaussianNB(**kwargs)

class SVM(BaseModel):

    def __init__(self, **kwargs) -> None:
        self.model = svm.SVC(**kwargs)
