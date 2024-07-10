import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV, Ridge, RidgeCV, ridge_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from code.em_algorithm import EMAlgorithm
from code.lasso import ElasticNet, lambda_search
from code.utils import cv_boxplot, generate_data, load_data, plot_E_beta, plot_E_beta_2, plot_E_beta_2_diag, split_train_test, cross_validate
from code.variational_inference import VariationalInference


def em_algorithm(y, Z, X, omega_init, sigma_b2_init, sigma_e2_init, max_iter=200, tol=1e-6, cv_folds=5):

    # ---------------- EM algorithm ----------------
    em_params = {'max_iter': max_iter, 'tol': tol, 'omega': omega_init, 'sigma_b2': sigma_b2_init, 'sigma_e2': sigma_e2_init}
    em = EMAlgorithm(**em_params)

    log_marginal_likelihoods, Theta, E_beta = em.fit(y, Z, X)

    print("Log marginal likelihoods:", log_marginal_likelihoods)
    print("Estimate of Theta:", Theta)
    # print("Posterior mean of beta:", E_beta)

    y_train_pred = em.predict(Z, X)
    print('EM Train MSE:', mean_squared_error(y, y_train_pred))

    # Plot marginal likelihood
    em.plot_marginal_likelihood('outputs/log_marginal_likelihoods.png')

    # Cross validation
    cv_mses = cross_validate(y, Z, X, EMAlgorithm, em_params, n_splits=cv_folds)

    print('CV Mean MSE:', np.mean(cv_mses))
    print('CV MSEs:', cv_mses)

    np.savetxt('outputs/em_cv_mses.csv', cv_mses, delimiter=',')


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
    vi.plot_marginal_likelihood_and_elbo('outputs/log_marginal_likelihoods_vi.png')
    vi.plot_gap('outputs/gap.png')

    # Cross validation
    # cv_mses = cross_validate(y, Z, X, VariationalInference, vi_params, n_splits=cv_folds)

    # print('CV Mean MSE:', np.mean(cv_mses))
    # print('CV MSEs:', cv_mses)

    # np.savetxt('outputs/vi_cv_mses.csv', cv_mses, delimiter=',')

    

def lasso(y, Z, X, cv_folds=10):

    # ---------------- LASSO -----------------
    lasso_params = {'alpha': 0.01227734, 'fit_intercept': False, 'l1_ratio': 1.0, 'max_iter': 1000}
    lasso = ElasticNet(**lasso_params)
    
    lasso.fit(y, Z, X)
    y_pred = lasso.predict(Z, X)

    print('LASSO Train MSE:', mean_squared_error(y, y_pred)) # Calculate MSE
    
    omega_beta = lasso.coef_
    omega = omega_beta[:Z.shape[1]]
    beta = omega_beta[Z.shape[1]:]

    np.save('outputs/beta_lasso.npy', beta)

    print("Estimated omega:", omega)
    print("Estimated beta:", beta)

    # Search for the best lambda
    lambdas = np.arange(-8, 0, 0.1)
    lambdas = np.exp(lambdas)
    min_mse, min_average_row = lambda_search(y, Z, X, ElasticNet, lasso_params, lambda_list = lambdas, save_path='outputs/lasso_cv_mses_all.csv', cv_folds=cv_folds)
    print("Minimum MSE:", min_mse)
    print("Minimum average row:", min_average_row)

    np.savetxt('outputs/lasso_cv_mses.csv', min_average_row[1:], delimiter=',')


if __name__ == '__main__':

    y, Z, X = load_data(path='data/lmm_y_z_x.txt')

    # ---------------- EM -----------------

    # em_algorithm(y, Z, X, omega_init=np.zeros(10), sigma_b2_init=1, sigma_e2_init=1, max_iter=1000, tol=1e-6, cv_folds=10)

    # plot_E_beta_2(data_path='outputs/E_beta_2_em.npy', save_path='outputs/E_beta_2_em.png')
    # plot_E_beta(data_path='outputs/E_beta_em.npy', save_path='outputs/E_beta_em.png')
    # plot_E_beta_2_diag(data_path='outputs/E_beta_2_em.npy', save_path='outputs/E_beta_2_diag_em.png')

    # ---------------- Lasso -----------------

    lasso(y, Z, X, cv_folds=10)

    plot_E_beta(data_path='outputs/E_beta_vi.npy', save_path='outputs/E_beta_vi.png')

    # cv_boxplot(cv_mses_dirs=['outputs/em_cv_mses.csv', 'outputs/lasso_cv_mses.csv'], save_path='outputs/cv_boxplot.png', labels=['EM', 'LASSO'])    

    # ---------------- VI -----------------

    # variational_inference(y, Z, X, omega_init=np.zeros(10), sigma_b2_init=0.8, sigma_e2_init=0.2, max_iter=100, tol=1e-6, cv_folds=10)

    # plot_E_beta(data_path='outputs/E_beta_vi.npy', save_path='outputs/E_beta_vi.png')
    # plot_E_beta_2_diag(data_path='outputs/E_beta_2_diag_vi.npy', save_path='outputs/E_beta_2_diag_vi.png')

    cv_boxplot(cv_mses_dirs=['outputs/em_cv_mses.csv', 'outputs/lasso_cv_mses.csv', 'outputs/vi_cv_mses.csv'], save_path='outputs/cv_boxplot_2.png', labels=['EM', 'LASSO', 'VI'])

    

    