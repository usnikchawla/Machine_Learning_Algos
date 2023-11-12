import numpy as np
class bayes():
    
    def __init__(self, x, y):
        
        self.x = x
        self.y = y
        self.cls = np.unique(self.y)
        self.thetas = []
        
    def train(self):
        
        for i in range(len(self.cls)):
            mean = self.x[self.y == i].mean(0)
            C = self.x[self.y == i] - mean
            S = np.cov(C.T)
            try:
                self.thetas.append([mean, np.linalg.det(S), np.linalg.pinv(S)])
            except:
                S = np.matrix(S)
                self.thetas.append([mean, np.linalg.det(S), np.linalg.pinv(S)])
            
    def test(self, x):
        
        values = []
        for i in range(len(self.cls)):
            mu, det, siginv = self.thetas[i]
            p = 1/(det + 1e-12)**0.5 
            p *= np.exp(-0.5 * np.matmul(np.matmul((x-mu), siginv), (x-mu).T))
            values.append(np.asarray(p).flatten()[0])
        return np.argmax(np.stack(values))
    
    def score(self, x, y):
        
        count = 0
        correct = 0
        for i in range(len(x)):
            y_ = self.test(x[i])
            if y[i] == y_:
                correct += 1
            count += 1
            
        return correct/count