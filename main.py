import numpy as np
from sklearn.linear_model import Ridge, ridge_regression
from sklearn.metrics import mean_squared_error
from code.cross_validation import cross_validate
from code.em_algorithm import EMAlgorithm
from code.lasso import ElasticNet, lambda_search
from code.load_data import generate_data, load_data, split_train_test
from code.variational_inference import VariationalInference


def em_algorithm(y, Z, X, omega_init, sigma_b2_init, sigma_e2_init, max_iter=200, tol=1e-6, cv_folds=5):

    y_train, Z_train, X_train, y_test, Z_test, X_test = split_train_test(y, Z, X, test_size=0.2)

    # ---------------- EM algorithm ----------------
    em_params = {'max_iter': max_iter, 'tol': tol, 'omega': omega_init, 'sigma_b2': sigma_b2_init, 'sigma_e2': sigma_e2_init}
    em = EMAlgorithm(**em_params)

    log_marginal_likelihoods, Theta, E_beta = em.fit(y_train, Z_train, X_train)

    print("Log marginal likelihoods:", log_marginal_likelihoods)
    print("Estimate of Theta:", Theta)
    # print("Posterior mean of beta:", E_beta)

    y_train_pred = em.predict(Z_train, X_train)
    print('EM Train MSE:', mean_squared_error(y_train, y_train_pred))
    y_test_pred = em.predict(Z_test, X_test)
    print('EM Test MSE:', mean_squared_error(y_test, y_test_pred))

    # Plot marginal likelihood
    em.plot_marginal_likelihood('outputs/log_marginal_likelihoods.png')

    # Cross validation
    # cv_mses = cross_validate(y, Z, X, EMAlgorithm, em_params, n_splits=cv_folds)

    # print('CV Mean MSE:', np.mean(cv_mses))
    # print('CV MSEs:', cv_mses)

    # np.save('outputs/em_cv_mses.npy', cv_mses)


def variational_inference(y, Z, X, omega_init, sigma_b2_init, sigma_e2_init, max_iter=200, tol=1e-6, cv_folds=5):

    y_train, Z_train, X_train, y_test, Z_test, X_test = split_train_test(y, Z, X, test_size=0.2)

    # ---------------- EM algorithm ----------------
    vi_params = {'max_iter': max_iter, 'tol': tol, 'omega': omega_init, 'sigma_b2': sigma_b2_init, 'sigma_e2': sigma_e2_init}
    vi = VariationalInference(**vi_params)

    log_marginal_likelihoods, Theta = vi.fit(y_train, Z_train, X_train)

    print("Log marginal likelihoods:", log_marginal_likelihoods)
    print("Estimate of Theta:", Theta)

    y_train_pred = vi.predict(Z_train, X_train)
    print('EM Train MSE:', mean_squared_error(y_train, y_train_pred))
    y_test_pred = vi.predict(Z_test, X_test)
    print('EM Test MSE:', mean_squared_error(y_test, y_test_pred))

    # Plot marginal likelihood
    vi.plot_marginal_likelihood('outputs/log_marginal_likelihoods_vi.png')

    # Cross validation
    # cv_mses = cross_validate(y, Z, X, EMAlgorithm, em_params, n_splits=cv_folds)

    # print('CV Mean MSE:', np.mean(cv_mses))
    # print('CV MSEs:', cv_mses)

    # np.save('outputs/em_cv_mses.npy', cv_mses)
    

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
    min_mse, min_average_row = lambda_search(y, Z, X, ElasticNet, lasso_params, lambda_list = np.arange(0.003, 0.004 + 0.0001, 0.0001), save_path='outputs/cv_mses_all.npy', cv_folds=cv_folds)
    print("Minimum MSE:", min_mse)
    print("Minimum average row:", min_average_row)


if __name__ == '__main__':
    omega = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
    # omega = omega / 10.0
    y, Z, X = generate_data(n = 10000, c = 10, p = 1000, omega=omega, sigma_b2 = 1, sigma_e2 = 1)
    # y, Z, X = load_data(path='data/lmm_y_z_x.txt')

    

    # em_algorithm(y, Z, X, omega_init=np.zeros(10), sigma_b2_init=0.5, sigma_e2_init=0.5, max_iter=200, tol=1e-6, cv_folds=10)

    # lasso(y, Z, X, cv_folds=5)

    # Ridge Regression
    # X = np.hstack((Z, X))
    # alpha = 0.01
    # ridge = Ridge(alpha=alpha)
    # ridge.fit(X, y)
    # print(ridge.coef_[:10])


    variational_inference(y, Z, X, omega_init=np.zeros(10), sigma_b2_init=0.5, sigma_e2_init=0.5, max_iter=200, tol=1e-6, cv_folds=5)

    