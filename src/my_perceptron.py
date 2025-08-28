import numpy as np

class Perceptron:
    def __init__(self,w=None,b=None):
        self.w = np.random.rand(2) if w is None else w
        self.b = np.random.rand(1) if b is None else b

    def classify(self,x):
        return np.sign(x@self.w + self.b)

    def fit(self,amostra,epoch_size=1000,lr=.005):
        for _ in range(epoch_size): 
            for ponto,gabarito in amostra:
                candidato = self.classify(ponto)
                
                erro = gabarito - candidato

                self.w += lr*erro*ponto
                self.b += lr*erro        


def gera_amostra(tamanho,desvio_padrao):
    pts_0 = []
    pts_1 = []
    for _ in range(tamanho):
        point_0 = np.random.normal(loc=np.array([0,0]),size=2,scale=desvio_padrao)
        pts_0.append((point_0,-1))
        
        point_1 = np.random.normal(loc=np.array([1,1]),size=2,scale=desvio_padrao)
        pts_1.append((point_1,1))
    amostra = pts_0 + pts_1
    return amostra

def compute_confusion_matrix(amostra,perceptron):
    VP,FP,FN,VN = 0,0,0,0
    for ponto,gabarito in amostra:
        resultado = perceptron.classify(ponto)
        if gabarito == 1:
            if resultado == 1:
                VP += 1
            else:
                FP += 1
        else:
            if resultado == -1:
                VN += 1
            else:
                FN += 1
    return (VP,FP,FN,VN)

def accuracy_score(confusion_matrix):
    VP,FP,FN,VN = confusion_matrix
    return (VP+VN)/(VP+FP+FN+VN)

def precision_score(confusion_matrix):
    VP,FP,FN,VN = confusion_matrix
    return VP/(VP+FP)

def recall_score(confusion_matrix):
    VP,FP,FN,VN = confusion_matrix
    return VP/(VP+FN)


def f1_score(confusion_matrix):
    precision = precision_score(confusion_matrix)
    recall = recall_score(confusion_matrix)
    return 2*(precision*recall)/(precision+recall)



