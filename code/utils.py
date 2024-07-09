from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def check_numpy_array(array):
    array_pd = pd.DataFrame(array)
    print(array_pd)
    print(array_pd.describe())

def load_data(path='lmm_y_z_x.txt'):
    data = np.loadtxt(path, delimiter='\t', skiprows=1)

    # check_numpy_array(data)
    # data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    y = data[:, 0]
    Z = data[:, 1:11]
    X = data[:, 11:]
    # sample p columns from X
    # X = X[:, np.random.choice(X.shape[1], 100, replace=False)]

    return y, Z, X

def generate_data(n=10000, c=10, p= 5000, omega=np.random.randn(10), sigma_b2 = 1, sigma_e2 = 0.5 ** 2):
    Z = np.random.randn(n, c)
    omega = np.array(omega)
    X = np.random.randn(n, p)

    beta = np.random.normal(0, np.sqrt(sigma_b2), p)
    e = np.random.normal(0, np.sqrt(sigma_e2), n)

    # print(beta)

    y = Z @ omega + X @ beta + e

    return y, Z, X

def split_train_test(y, Z, X, test_size=0.2):
    n = y.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_indices = indices[: int(n * (1 - test_size))]
    test_indices = indices[int(n * (1 - test_size)) :]

    y_train = y[train_indices]
    Z_train = Z[train_indices, :]
    X_train = X[train_indices, :]
    y_test = y[test_indices]
    Z_test = Z[test_indices, :]
    X_test = X[test_indices, :]

    return y_train, Z_train, X_train, y_test, Z_test, X_test


def cross_validate(y, Z, X, model_class, model_params, n_splits=5):
    # seperate data into train and test
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    mses = np.zeros(n_splits)
    
    for i, (train_index, test_index) in enumerate(kf.split(y)):
        y_train, y_test = y[train_index], y[test_index]
        Z_train, Z_test = Z[train_index], Z[test_index]
        X_train, X_test = X[train_index], X[test_index]
        # fit model
        model = model_class(**model_params)
        model.fit(y_train, Z_train, X_train)
        # Predict test set
        y_pred = model.predict(Z_test, X_test)
        # Calculate MSE
        mses[i] = mean_squared_error(y_pred, y_test)
    
    return mses


def cv_boxplot_2(cv_mses, save_path='boxplot.png', labels=['Lasso', 'EM', 'VI']):
    """
    Plot boxplot of cross-validation results
    """

    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.boxplot(cv_mses, labels=labels)
    ax.set_ylabel('MSE')
    ax.set_title('10-fold Cross Validation Results')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def cv_boxplot(cv_mses_dirs, save_path='cv_boxplot.png', labels=['EM', 'LASSO', 'VI']):
    cv_mses = []
    for i in range(len(cv_mses_dirs)):
        cv_mses_ = np.genfromtxt(cv_mses_dirs[i])
        cv_mses.append(cv_mses_)

    cv_boxplot_2(cv_mses, save_path, labels)