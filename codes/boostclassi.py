import numpy as np
    
class svm():
    
    def __init__(self, x, y, penalty=0.5):
        
        self.x = x
        self.y = y
        self.penalty = penalty
    
    def train(self, x, y, p, verbose=True, lr=0.1):
        
        N = len(x)
        mu = np.zeros(len(x[0]))
        b = 0
        
        def cost(mu, x, y, b, p):
            y[y==0] = -1
            c = (1 - y*(np.matmul(x, mu)+b))*p
            c[c<0] = 0
            return 0.5*np.matmul(mu, mu) + self.penalty*c.sum()
        
        def update(mu, x, y, b, p):
            
            y[y==0] = -1
            c = 1 - y*(np.matmul(x, mu)+b)
            c[c<0] = 0
            gradw = np.zeros(mu.shape) + mu
            gradb = 0
            
            for i in range(len(c)):                
                if c[i] != 0:
                    gradw += -self.penalty*x[i]*y[i]*p[i]
                    gradb += -self.penalty*y[i]*p[i]
                    
            return mu - lr*gradw, b - lr*gradb
        
        i = 0
        prev = 1
        while True:
            
            if verbose == True:
                print("Itr:{}, Cost:{}".format(i, prev), flush=True, end='\r')
                
            mu, b = update(mu, x, y, b, p)
            tmp = cost(mu, x, y, b, p)
            self.mu = mu
            self.b = b           
            if i >= 20:
                break
                
            i += 1
            prev = tmp 
        
    def test(self, x):

        y = 0
        for i in range(len(x)):
            y += self.mu[i] * x[i]
        y += self.b
        
        return y
    
    def score(self, x, y, p):
        
        count = 0
        correct = 0
        wrong = 0
        y[y==0] = -1
        sign = []
        
        for i in range(len(x)):
            y_ = int(np.sign(self.test(x[i])))
            sign.append(y_)
            
            if y[i] == y_:
                correct += 1
                
            else:
                wrong += 1*p[i]
                
            count += 1
            
        return wrong , correct/count, np.stack(sign)
    
def adaboost(x, y, steps, penalty=1, lr=0.1):
 
    models = []
    A = []
    eps = []
    scores = []
    w = [1/len(x) for i in range(len(x))]
    y[y == 0] = -1

    for i in range(steps):
        
        print("{}/{}".format(i+1, steps), flush=True, end='\r')
        p = [w[j]/sum(w) for j in range(len(w))]
        
        model = svm(x=x, y=y, penalty=penalty)
        model.train(x, y, p, False, lr)
        ep, score, sign = model.score(x, y, p)

        try:
            assert ep < 0.5, "Epsilon_i greater than 0.5"
        except:
            return models, A
        
        a = 0.5*np.log((1-ep)/ep)
        
        eps.append(ep)
        scores.append(score)
        models.append(model)
        A.append(a)
        
        w = [w[j]*np.exp(-a*sign[j]*y[j]) for j in range(len(w))]
        
    return models, A