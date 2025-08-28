import numpy as np

class Perceptron:
    def __init__(self,w=None,b=None,random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
            
        self.w = np.random.rand(2) if w is None else w
        self.b = np.random.rand(1) if b is None else b

    def classify(self,x):
        return np.sign(x@self.w + self.b)

    def fit(self,samples,classifications,epoch_size=1000,lr=.005):
        for _ in range(epoch_size): 
            for sample,classification in zip(samples,classifications):
                result = self.classify(sample)
                
                err = classification - result

                self.w += lr*err*sample
                self.b += lr*err    

    def score(self,samples,classifications):
        TP,FP,FN,TN = 0,0,0,0
        for sample,classification in zip(samples,classifications):
            result = self.classify(sample)
            if classification == 1:
                if result == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if result == -1:
                    TN += 1
                else:
                    FN += 1
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1_score = 2*(precision*recall)/(precision+recall)
        return f1_score



