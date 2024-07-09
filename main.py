import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV, Ridge, RidgeCV, ridge_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from code.em_algorithm import EMAlgorithm
from code.lasso import ElasticNet, lambda_search
from code.utils import cv_boxplot, generate_data, load_data, split_train_test, cross_validate
from code.variational_inference import VariationalInference


def em_algorithm(y, Z, X, omega_init, sigma_b2_init, sigma_e2_init, max_iter=200, tol=1e-6, cv_folds=5):

    # y_train, Z_train, X_train, y_test, Z_test, X_test = split_train_test(y, Z, X, test_size=0.2)

    y_train, Z_train, X_train = y, Z, X

    # ---------------- EM algorithm ----------------
    em_params = {'max_iter': max_iter, 'tol': tol, 'omega': omega_init, 'sigma_b2': sigma_b2_init, 'sigma_e2': sigma_e2_init}
    em = EMAlgorithm(**em_params)

    log_marginal_likelihoods, Theta, E_beta = em.fit(y_train, Z_train, X_train)

    print("Log marginal likelihoods:", log_marginal_likelihoods)
    print("Estimate of Theta:", Theta)
    # print("Posterior mean of beta:", E_beta)

    y_train_pred = em.predict(Z_train, X_train)
    print('EM Train MSE:', mean_squared_error(y_train, y_train_pred))
    # y_test_pred = em.predict(Z_test, X_test)
    # print('EM Test MSE:', mean_squared_error(y_test, y_test_pred))

    # Plot marginal likelihood
    em.plot_marginal_likelihood('outputs/log_marginal_likelihoods.png')

    # Cross validation
    # cv_mses = cross_validate(y, Z, X, EMAlgorithm, em_params, n_splits=cv_folds)

    # print('CV Mean MSE:', np.mean(cv_mses))
    # print('CV MSEs:', cv_mses)

    # np.savetxt('outputs/em_cv_mses.csv', cv_mses, delimiter=',')


def variational_inference(y, Z, X, omega_init, sigma_b2_init, sigma_e2_init, max_iter=200, tol=1e-6, cv_folds=5):

    # ---------------- EM algorithm ----------------
    vi_params = {'max_iter': max_iter, 'tol': tol, 'omega': omega_init, 'sigma_b2': sigma_b2_init, 'sigma_e2': sigma_e2_init}
    vi = VariationalInference(**vi_params)

    log_marginal_likelihoods, Theta = vi.fit(y, Z, X)
  
    print("Log marginal likelihoods:", log_marginal_likelihoods)
    print("Estimate of Theta:", Theta)

    y_train_pred = vi.predict(Z, X)
    print('EM Train MSE:', mean_squared_error(y, y_train_pred))

    # Plot marginal likelihood
    vi.plot_marginal_likelihood('outputs/log_marginal_likelihoods_vi.png')

    # Cross validation
    cv_mses = cross_validate(y, Z, X, VariationalInference, vi_params, n_splits=cv_folds)

    print('CV Mean MSE:', np.mean(cv_mses))
    print('CV MSEs:', cv_mses)

    np.savetxt('outputs/vi_cv_mses.csv', cv_mses, delimiter=',')

    

def lasso(y, Z, X, cv_folds=10):

    # ---------------- LASSO -----------------
    lasso_params = {'alpha': 0.001, 'fit_intercept': False, 'l1_ratio': 1.0, 'max_iter': 1000}
    lasso = ElasticNet(**lasso_params)
    
    lasso.fit(y, Z, X)
    y_pred = lasso.predict(Z, X)

    print('LASSO Train MSE:', mean_squared_error(y, y_pred)) # Calculate MSE
    
    omega_beta = lasso.coef_
    omega = omega_beta[:Z.shape[1]]
    beta = omega_beta[Z.shape[1]:]

    print("Estimated omega:", omega)
    print("Estimated beta:", beta)

    # Search for the best lambda
    lambdas = np.arange(-7, -2, 0.1)
    lambdas = np.exp(lambdas)
    min_mse, min_average_row = lambda_search(y, Z, X, ElasticNet, lasso_params, lambda_list = lambdas, save_path='outputs/lasso_cv_mses_all.csv', cv_folds=cv_folds)
    print("Minimum MSE:", min_mse)
    print("Minimum average row:", min_average_row)

    np.savetxt('outputs/lasso_cv_mses.csv', min_average_row[1:], delimiter=',')


if __name__ == '__main__':
    omega = np.array([-1, 2, -3, 1, -2, 3, -1, 2, -3, 1])
    # omega = omega / 100.0
    # omega = [-4.34232671e-04, -4.36072309e-02, -5.21779989e-02, -1.10921390e-02, 5.28039854e-02, 2.24489587e-02, 5.11428609e-02, -1.37524736e-02, 2.27649024e-02, -3.88254831e-02]
    # y, Z, X = generate_data(n = 2000, c = 10, p = 500, omega=omega, sigma_b2 = 0.25, sigma_e2 = 0.75)
    y, Z, X = load_data(path='data/lmm_y_z_x.txt')


    # em_algorithm(y, Z, X, omega_init=np.zeros(10), sigma_b2_init=0.5, sigma_e2_init=0.5, max_iter=1000, tol=1e-2, cv_folds=10)


    # lasso(y, Z, X, cv_folds=10)

    # cv_boxplot(cv_mses_dirs=['outputs/em_cv_mses.csv', 'outputs/lasso_cv_mses.csv'], save_path='outputs/cv_boxplot.png', labels=['EM', 'LASSO'])    


    variational_inference(y, Z, X, omega_init=np.zeros(10), sigma_b2_init=1, sigma_e2_init=0.5, max_iter=100, tol=1e-6, cv_folds=10)

    cv_boxplot(cv_mses_dirs=['outputs/em_cv_mses.csv', 'outputs/lasso_cv_mses.csv', 'outputs/vi_cv_mses.csv'], save_path='outputs/cv_boxplot_2.png', labels=['EM', 'LASSO', 'VI'])

    