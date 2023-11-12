import scipy.io
import numpy as np

def get_data(file='poses'):

    if file == 'data':
        data = scipy.io.loadmat('C:\\Users\\HP\Desktop\\117377460_P1\\codes\\data\\data.mat')['face']
        cls_label = np.arange(200)
        cls_label = np.repeat(cls_label, 3)
        exp_label = np.stack([0, 1, 2]*200)
        data = np.transpose(data, (2, 0, 1))
        return data, cls_label, exp_label
    
    else:
        data = scipy.io.loadmat('C:\\Users\\HP\Desktop\\117377460_P1\\codes\\data\\pose.mat')['pose']
        data = data.reshape((48*40, 13, 68))
        data = np.concatenate(np.transpose(data, (2, 1, 0)), axis=0)
        cls_label = np.arange(68)
        cls_label = np.repeat(cls_label, 13)
        exp_label = cls_label        
        return data, cls_label, exp_label

def split_data(data, cls_label, exp_label, fraction, split='poses'):
    
    assert split in ['poses', 'expression']
    inds = []
    
    if split == 'poses':
        
        select = int(fraction*68*13)
        ind = np.random.choice(np.arange(68*13), size=select, replace=False)
        for i in ind:
            inds.append(i)
        inds = np.stack(inds)
        
        
    else:
        
        select = int(fraction*400)
        
        # remove illuminations
        ind = (exp_label != 2)
        data = data[ind]
        
        cls_label = cls_label[ind]
        exp_label = exp_label[ind]
        
        ind = np.random.choice(np.arange(400), size=select, replace=False)
        for i in ind:
            inds.append(i)
        inds = np.stack(inds)
                     
    test_inds = np.setdiff1d(np.arange(len(data)), inds)
    train_x, test_x = data[inds], data[test_inds]
    train_slabel, test_slabel = cls_label[inds], cls_label[test_inds]
    train_elabel, test_elabel = exp_label[inds], exp_label[test_inds]   
    
    return train_x, train_slabel, train_elabel, test_x, test_slabel, test_elabel