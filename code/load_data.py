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