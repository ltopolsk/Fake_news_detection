import pickle
from sklearn import naive_bayes, svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

class BaseModel:

    def __init__(self, **kwargs) -> None:
        self.model = None

    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path):
        self.model = pickle.load(open(path, 'rb'))

    def train(self, data, labels, **kwargs):
        self.model.fit(data, labels)

    def eval(self, data, labels):
        labels_pred = self.model.predict(data)
        return {
            'accuracy':accuracy_score(labels_pred, labels),
            'precision':precision_score(labels, labels_pred),
            'recall':recall_score(labels, labels_pred,),
            'f1_score':f1_score(labels, labels_pred,),
            'confusion_matrix':confusion_matrix(labels, labels_pred,)
        }

class NaiveBayes(BaseModel):

    def __init__(self, **kwargs) -> None:
        self.model = naive_bayes.GaussianNB(**kwargs)

class SVM(BaseModel):

    def __init__(self, **kwargs) -> None:
        self.model = svm.SVC(**kwargs)
