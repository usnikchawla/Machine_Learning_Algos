from data import get_data, split_data
from functions import mda, pca
from bclassi import bayes
import numpy as np
from matplotlib import pyplot as plt

train_acc = []
test_acc = []

for k in [5, 10]:

    split = 'poses'
    fraction = 0.8

    np.random.seed(42)
    data, sub_label, exp_label = get_data()
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
    # A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    model = bayes(train_x, train_y)
    model.train()

    train_acc.append(model.score(train_x, train_y))
    test_acc.append(model.score(test_x, test_y))
    
plt.plot(['5', '10'], train_acc, label='Train data')
plt.plot(['5', '10'], test_acc, label='Test data')
plt.legend()
plt.xlabel('component dimension from 504 to x')
plt.ylabel('Accuracy')
plt.title('MDA + Bayes + poses classification')
plt.show()


train_acc = []
test_acc = []

for k in [5, 10, 50]:

    split = 'poses'
    fraction = 0.8

    np.random.seed(42)
    data, sub_label, exp_label = get_data()
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

    # A = mda(train_x, train_y, k)
    A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    model = bayes(train_x, train_y)
    model.train()

    train_acc.append(model.score(train_x, train_y))
    test_acc.append(model.score(test_x, test_y))

plt.plot(['5', '10', '50'], train_acc, '-o', label='Train data')
plt.plot(['5', '10', '50'], test_acc, '-o', label='Test data')
plt.legend()
plt.xlabel('component dimension from 504 to x')
plt.ylabel('Accuracy')
plt.title('PCA + Bayes + pose classification')
plt.show()