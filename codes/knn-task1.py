from data import get_data, split_data
from functions import mda, pca
from kclassi import knn
import numpy as np
from matplotlib import pyplot as plt

for k in [5, 10]:
    train_acc = []
    test_acc = []
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
    
    for nn in [1 ,5, 10, 20, 50]:
        model = knn(train_x, train_y, nn)
        train_acc.append(model.score(train_x, train_y))
        test_acc.append(model.score(test_x, test_y))
        
        
    plt.plot(['1', '5', '10', '20', '50'], train_acc, label='Train data.  reduced dimension = {}'.format(k))
    plt.plot(['1', '5', '10', '20', '50'], test_acc, label='Test data. reduced dimanesion = {}'.format(k))
    

plt.legend()
plt.xlabel('#  K nearest neighbors')
plt.ylabel('Accuracy')
plt.title('MDA + KNN + pose classification')
plt.show()


for k in [5, 10, 50]:
    train_acc = []
    test_acc = []
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

    
    A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))
    
    for nn in [1 ,5, 10, 20, 50]:
        model = knn(train_x, train_y, nn)
        train_acc.append(model.score(train_x, train_y))
        test_acc.append(model.score(test_x, test_y))
        print(nn)
        
    plt.plot(['1', '5', '10', '20', '50'], train_acc, label='Train data.  reduced dimension = {}'.format(k))
    plt.plot(['1', '5', '10', '20', '50'], test_acc, label='Test data. reduced dimanesion = {}'.format(k))
    

plt.legend()
plt.xlabel('#  K nearest neighbors')
plt.ylabel('Accuracy')
plt.title('PCAA + KNN + pose classification')
plt.show()