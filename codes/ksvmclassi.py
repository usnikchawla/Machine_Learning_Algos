import numpy as np
class kernel_svm():
    
    def __init__(self, x, y, kernel='exp', param=2, penalty=0.5, lr=0.1):
        
        self.x = x
        self.y = y
        self.kernel = kernel
        self.param = param
        self.penalty = penalty
        self.lr = lr
        
    def exp_kernel(self, x, y, param):        
        return np.exp(-np.linalg.norm(x-y)**2/param**2)
    
    def poly_kernel(self, x, y, param):
        return (1 + np.matmul(x, y.T))**param
    
    def train(self, x, y, verbose=True):
        
        N = len(x)
        K = np.zeros((N, N))
        mu = np.random.rand(N) 
        b = 0
            
        if self.kernel == 'exp':
            kernel = self.exp_kernel
        else:
            kernel = self.poly_kernel

        for i in range(N):
            for j in range(N):
                d = kernel(x[i], x[j], self.param)
                K[i][j] = d
        
        def cost(mu, K, y, b):
            y[y==0] = -1
            c = 1 - y*(np.matmul(K, mu)+b)
            c[c<0] = 0
            return 0.5*np.matmul(mu, mu) + c.sum()*self.penalty
        
        def update(mu, K, y, b):
            
            y[y==0] = -1
            c = 1 - y*(np.matmul(K, mu)+b)
            c[c<0] = 0
            gradw = np.zeros(mu.shape) + mu
            gradb = 0
            
            for i in range(len(c)):                
                if c[i] != 0:
                    gradw += -K[i]*y[i]*self.penalty
                    gradb += -y[i]*self.penalty
                    
            gradw /= len(K)
            gradb /= len(K)
            return mu - self.lr*gradw, b - self.lr*gradb
        
        i = 0
        prev = 100
        while True:
            if verbose == True:
                print("Itr:{}, Cost:{}".format(i, prev), flush=True, end='\r')
            mu, b = update(mu, K, y, b)
            tmp = cost(mu, K, y, b)
            if (prev - tmp < 1e-6 and i >= 100) or i >= 1000 :
                break
            i += 1
            prev = tmp
            
        self.mu = mu
        self.b = b                
        
    def test(self, x):

        if self.kernel == 'exp':
            kernel = self.exp_kernel
        else:
            kernel = self.poly_kernel

        y = 0
        for i in range(len(self.x)):
            y += self.mu[i] * kernel(self.x[i], x, self.param)
        y += self.b
        
        return y
    
    def score(self, x, y):
        
        count = 0
        correct = 0
        y[y==0] = -1
        for i in range(len(x)):
            y_ = int(np.sign(self.test(x[i])))
            if y[i] == y_:
                correct += 1
            count += 1
            
        return correct/count