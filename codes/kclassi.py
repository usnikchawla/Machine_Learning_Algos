import numpy as np
import statistics
from scipy import stats
    
class knn():
    
    def __init__(self, x, y, k):
        
        self.x = x
        self.y = y
        self.k = k
        
    def test(self, x):
        
        dist = []
        for i in range(len(self.x)):
            dist.append(np.linalg.norm(self.x[i] - x, ord=2))
        dist = np.stack(dist)
        inds = np.argsort(dist)[: self.k]
        
        labels = self.y[inds]
        return stats.mode(labels).mode[0]
    
    def score(self, x, y):
        
        count = 0
        correct = 0
        for i in range(len(x)):
            y_ = self.test(x[i])
            if y[i] == y_:
                correct += 1
            count += 1
            
        return correct/count 