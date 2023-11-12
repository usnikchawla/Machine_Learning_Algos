from numpy import linalg
import numpy as np

def mda(x, y, k=10):
    
    nums = np.unique(y)
    mu = []
    count = []
    
    for i in range(len(nums)):
        mu.append(x[y == nums[i]].mean(0))
        count.append((y == nums[i]).sum())
        
    mu = np.stack(mu)
    mu0 = np.zeros(mu[0].shape)
    for i in range(len(mu)):
        mu0 += count[i]*mu[i]
    mu0 /= sum(count)
    
    Sb = np.matrix(np.zeros((len(mu0), len(mu0))))
    for i in range(len(mu)):
        Sb += count[i]/sum(count) * np.matmul(np.matrix(mu[i]-mu0).T, np.matrix(mu[i]-mu0))
        
    Sw = np.matrix(np.zeros((len(mu0), len(mu0))))
    for i in range(len(x)):
        Sw += 1/sum(count) * np.matmul(np.matrix(x[i] - mu[y[i]]).T, np.matrix(x[i] - mu[y[i]]))
        
    A = np.matmul(np.linalg.pinv(Sw), Sb)
    _, C = linalg.eigh(A)
    
    C = C.T[-k:].T
    return C

def pca(x, k=10):
    
    A = np.cov(x.T)
    _, C = linalg.eigh(A)
    
    C = C.T[-k:].T
    return C