from data import get_data, split_data
from functions import mda, pca
from ksvmclassi import kernel_svm
import numpy as np
from matplotlib import pyplot as plt


def cross_validation(fold=3, kernel='exp', lr=0.5):

    N = len(train_x)
    n = int(N/fold)
    if kernel == 'exp':
        search_space = [i+6 for i in  range(15)]
    else:
        search_space = [i+1 for i in  range(15)]
    Score = 0

    for k in range(len(search_space)):
        score = 0
        j = search_space[k]
        print('Done {}/{}.'.format(k+1, len(search_space)), flush=True, end='\r')

        
        for i in range(fold):

            ind = np.arange(N)
            np.random.seed(i)
            np.random.shuffle(ind)

            x1 = train_x[ind[: (fold-1)*n]]
            y1 = train_y[ind[: (fold-1)*n]]
            x2 = train_x[ind[(fold-1)*n: ]]
            y2 = train_y[ind[(fold-1)*n: ]]
            
            model = kernel_svm(x=x1, y=y1, kernel=kernel, param=j, penalty=1, lr=lr)
            model.train(x1, y1, verbose=False)
            score += model.score(x2, y2)

        if score/fold > Score:
            Score = score/fold
            param = j

    model = kernel_svm(x=train_x, y=train_y, kernel=kernel, param=param, penalty=1, lr=lr)
    model.train(train_x, train_y)
    print('\nKernel parameter is', param, kernel)
    return model

train_acc = []
test_acc = []

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
    # A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    model = cross_validation(fold=3, kernel='exp', lr=0.1)

    train_acc.append(model.score(train_x, train_y))
    test_acc.append(model.score(test_x, test_y))
    print(k)
    
    
plt.plot(['5', '10'], train_acc, label='Train data')
plt.plot(['5', '10'], test_acc, label='Test data')
plt.legend()
plt.xlabel('# components after dimension reduction')
plt.ylabel('Accuracy')
plt.title('MDA + Kernel SVM (RBF) + Task 2')
plt.show()


train_acc = []
test_acc = []

for k in [5, 10, 50]:

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

    # A = mda(train_x, train_y, k)
    A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    model = cross_validation(fold=3, kernel='exp')

    train_acc.append(model.score(train_x, train_y))
    test_acc.append(model.score(test_x, test_y))
    print(k)
    

plt.plot(['5', '10', '50'], train_acc, label='Train data')
plt.plot(['5', '10', '50'], test_acc, label='Test data')
plt.legend()
plt.xlabel('# components after dimension reduction')
plt.ylabel('Accuracy')
plt.title('PCA + Kernel SVM (RBF) + Task 2')
plt.show()


# ============================ #

train_acc = []
test_acc = []

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
    # A = pca(train_x, k=k)

    train_x = np.asarray(np.matmul(train_x, A))
    test_x = np.asarray(np.matmul(test_x, A))

    model = cross_validation(fold=3, kernel='poly', lr=1)

    train_acc.append(model.score(train_x, train_y))
    test_acc.append(model.score(test_x, test_y))
    print(k)
    

plt.plot(['5', '10'], train_acc, '-o', label='Train data')
plt.plot(['5', '10'], test_acc, '-o', label='Test data')
plt.legend()
plt.xlabel('# components after dimension reduction')
plt.ylabel('Accuracy')
plt.title('MDA + Kernel SVM (poly) + Task 2')
plt.show()


train_acc = []
test_acc = []

for k in [5, 10, 50]:

    split = 'expression'
    fraction = 0.8

    np.random.seed(42)
    data, sub_label, exp_label = get_data('data')
    train_x, train_slabel, train_elabel, test_x, test_slabel, test_elabel = split_data(data, sub_label, exp_label, fraction, split=split)

    train_x = train_x.reshape((len(train_x), -1))
    test_x = test_x.reshape((len(test_x), -1))

    if split == 'subject':    
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

    model = cross_validation(fold=3, kernel='poly')

    train_acc.append(model.score(train_x, train_y))
    test_acc.append(model.score(test_x, test_y))
    print(k)
    
plt.plot(['5', '10', '50'], train_acc, label='Train data')
plt.plot(['5', '10', '50'], test_acc, label='Test data')
plt.legend()
plt.xlabel('# components after dimension reduction')
plt.ylabel('Accuracy')
plt.title('PCA + Kernel SVM (poly) + Task 2')
plt.show()
