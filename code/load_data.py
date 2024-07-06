import numpy as np


def load_data(path='lmm_y_z_x.txt'):
    data = np.loadtxt(path, delimiter='\t', skiprows=1)
    y = data[:, 0]
    Z = data[:, 1:11]
    X = data[:, 11:]

    return y, Z, X

def generate_data(n=10000, c=10, p= 5000, omega=np.random.randn(10), sigma_b2 = 1, sigma_e2 = 0.5 ** 2):
    Z = np.random.randn(n, c)
    omega = np.array(omega)
    X = np.random.randn(n, p)

    beta = np.random.normal(0, np.sqrt(sigma_b2), p)
    e = np.random.normal(0, np.sqrt(sigma_e2), n)

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

