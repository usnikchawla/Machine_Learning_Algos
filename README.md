
# Project: Bayesian, k-NN, Kernel SVM, and Boosted Linear SVM Classifiers

## Overview

This project involves implementing and evaluating multiple classifiers including Bayesian Classifier, k-NN, Kernel SVM, and Boosted Linear SVM. The classifiers are applied to two datasets (`pose.mat` for task 1 and `data.mat` for task 2), and the results are visualized using plots generated during the execution of the scripts.

### Classifiers Implemented:
- **Bayesian Classifier**: `bayes-task1.py`, `bayes-task2.py`
- **k-Nearest Neighbors (k-NN)**: `knn-task1.py`, `knn-task2.py`
- **Kernel Support Vector Machine (SVM)**: `kernelsvm-task2.py`
- **Boosted Linear SVM (AdaBoost)**: `adaboost-task2.py`

---

## Requirements

### System Requirements:
- Python 3.9.6 (GCC 7.5.0 Anaconda on Linux)

### Python Dependencies:
- **NumPy**: 1.20.3
- **SciPy**: 1.7.1
- **Seaborn**: 0.11.2
- **Matplotlib**: 3.4.2

You can install the required dependencies using `pip`:
```bash
pip install numpy==1.20.3 scipy==1.7.1 seaborn==0.11.2 matplotlib==3.4.2
```

---

## Data

- **Task 1**: The dataset is stored in `codes/data/pose.mat`.
- **Task 2**: The dataset is stored in `codes/data/data.mat`.

Ensure the paths to the datasets are correctly set in the scripts for proper loading and processing.

---

## Folder Structure

```
codes/
│
├── data/
│   ├── data.mat         # Dataset for Task 2
│   └── pose.mat         # Dataset for Task 1
│
├── data.py              # Script for reading, processing, and partitioning the data
├── functions.py         # Preprocessing functions: PCA and MDA
│
├── bayes-task1.py       # Bayesian Classifier for Task 1
├── bayes-task2.py       # Bayesian Classifier for Task 2
│
├── knn-task1.py         # k-NN Classifier for Task 1
├── knn-task2.py         # k-NN Classifier for Task 2
│
├── kernelsvm-task2.py   # Kernel SVM Classifier for Task 2
└── adaboost-task2.py    # Boosted Linear SVM Classifier for Task 2
```

---

## Running the Scripts

1. **Bayesian Classifier:**
    - **Task 1**: 
      ```bash
      cd codes/
      python bayes-task1.py
      ```
    - **Task 2**:
      ```bash
      cd codes/
      python bayes-task2.py
      ```
    The output plots will be displayed after execution.

2. **k-NN Classifier:**
    - **Task 1**:
      ```bash
      cd codes/
      python knn-task1.py
      ```
    - **Task 2**:
      ```bash
      cd codes/
      python knn-task2.py
      ```

3. **Kernel SVM Classifier (Task 2)**:
    ```bash
    cd codes/
    python kernelsvm-task2.py
    ```

4. **Boosted Linear SVM Classifier (Task 2)**:
    ```bash
    cd codes/
    python adaboost-task2.py
    ```

---

## Scripts Overview

- **data.py**: 
    - Reads, processes, and partitions the data from `.mat` files. Modify the paths in this file according to your local machine if necessary.

- **functions.py**: 
    - Contains data preprocessing functions such as Principal Component Analysis (PCA) and Multiple Discriminant Analysis (MDA).

- **bayes-task1.py** and **bayes-task2.py**: 
    - These scripts implement and run the Bayesian classifier for Task 1 and Task 2, respectively. Running these scripts will process the data, apply the classifier, and display output plots.

- **knn-task1.py** and **knn-task2.py**: 
    - These scripts implement and run the k-Nearest Neighbors classifier for Task 1 and Task 2, respectively. Running these scripts will process the data, apply the classifier, and display output plots.

- **kernelsvm-task2.py**: 
    - This script implements the Kernel SVM classifier for Task 2.

- **adaboost-task2.py**: 
    - This script implements the Boosted Linear SVM (AdaBoost) classifier for Task 2.

---

## Author

- **Usnik Chawla** (uchawla@umd.edu)

