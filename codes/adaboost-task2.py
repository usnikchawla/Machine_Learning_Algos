from data import get_data, split_data
from functions import mda, pca
from boostclassi import adaboost
import numpy as np
from matplotlib import pyplot as plt


for k in [5, 10]:
    
    split = 'expression'
    fraction = 0.8

    np.random.seed(42)
    data, sub_label, exp_label = get_data('data')
    train_x, train_slabel, train_elabel, test_x, test_slabel, test_elabel = split_data(data, sub_label, exp_label, fraction, split=split)

    train_x = train_x.reshape((len(train_x), -1))
    test_x = test_x.reshape((len(test_x), -1))

    if split == 'poses':    
        train_y = train_slabel
        test_y = test_slabel

    else:
        train_y = train_elabel
        test_y = test_elabel

    mu = train_x.mean(0)
    std = train_x.std(0)
    train_x = (train_x-mu)/std
    test_x = (test_x-mu)/std

    A = mda(train_x, train_y, k)


    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    models, A = adaboost(train_x, train_y, 50, penalty=1, lr=0.75)

    results = np.zeros(len(train_x))
    train_y[train_y==0] = -1
    y = train_y
    results = np.zeros(len(train_x))

    train_acc = []
    test_acc = []

    for j in range(len(models)):

        p = np.ones(len(train_x))/len(train_x)
        _, _, signs= models[j].score(train_x, train_y, p)
        results += signs*A[j]    
        y_ = np.sign(results)
        acc = (y == y_).sum()/len(y)
        train_acc.append(acc)

    print()
    results = np.zeros(len(test_x))
    y = test_y
    y[y==0] = -1
    results = np.zeros(len(test_x))

    for j in range(len(models)):

        p = np.ones(len(test_x))/len(train_x)
        _, _, signs= models[j].score(test_x, test_y, p)
        results += signs*A[j]    
        y_ = np.sign(results)
        acc = (y == y_).sum()/len(y)
        test_acc.append(acc)
        


    plt.plot(train_acc, label='Train data. n_components = {}'.format(k),)
    plt.plot(test_acc, label='Test data. n_components = {}'.format(k),)
    
    
plt.legend()
plt.xlabel('# Boost step')
plt.ylabel('Accuracy')
plt.title('MDA + Boosted SVM +  Task 2')
plt.show()

for k in [5, 10, 15, 20]:
    
    split = 'expression'
    fraction = 0.8

    np.random.seed(42)
    data, sub_label, exp_label = get_data('data')
    train_x, train_slabel, train_elabel, test_x, test_slabel, test_elabel = split_data(data, sub_label, exp_label, fraction, split=split)

    train_x = train_x.reshape((len(train_x), -1))
    test_x = test_x.reshape((len(test_x), -1))

    if split == 'poses':    
        train_y = train_slabel
        test_y = test_slabel

    else:
        train_y = train_elabel
        test_y = test_elabel

    mu = train_x.mean(0)
    std = train_x.std(0)
    train_x = (train_x-mu)/std
    test_x = (test_x-mu)/std


    A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    models, A = adaboost(train_x, train_y, 50)

    results = np.zeros(len(train_x))
    train_y[train_y==0] = -1
    y = train_y
    results = np.zeros(len(train_x))

    train_acc = []
    test_acc = []

    for j in range(len(models)):

        p = np.ones(len(train_x))/len(train_x)
        _, _, signs= models[j].score(train_x, train_y, p)
        results += signs*A[j]    
        y_ = np.sign(results)
        acc = (y == y_).sum()/len(y)
        train_acc.append(acc)

    print()
    results = np.zeros(len(test_x))
    y = test_y
    y[y==0] = -1
    results = np.zeros(len(test_x))

    for j in range(len(models)):

        p = np.ones(len(test_x))/len(train_x)
        _, _, signs= models[j].score(test_x, test_y, p)
        results += signs*A[j]    
        y_ = np.sign(results)
        acc = (y == y_).sum()/len(y)
        test_acc.append(acc)
        

    
    plt.plot(train_acc, label='Train data. n_components = {}'.format(k))
    plt.plot(test_acc, label='Test data. n_components = {}'.format(k))
   
    
plt.legend()
plt.xlabel('# Boost step')
plt.ylabel('Accuracy')
plt.title('PCA + Boosted SVM + Task 2')
plt.show()

