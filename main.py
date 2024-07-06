import numpy as np
from sklearn.metrics import mean_squared_error
from code.em_algorithm import em_algorithm, plot_marginal_likelihood
from code.lasso import ElasticNet, lambda_search
from code.load_data import generate_data, load_data


if __name__ == '__main__':
    y, Z, X = generate_data(n = 10000, c = 10, p = 5000, omega=[1, 2, 3, 1, 2, 3, 1, 2, 3, 1], sigma_b2 = 1, sigma_e2 = 1)
    # y, Z, X = load_data(path='lmm_y_z_x.txt')

    # ---------------- EM algorithm ----------------
    log_marginal_likelihoods, Theta, E_beta = em_algorithm(y, Z, X, tol=1e-6, max_iter=1000)
    print("Log marginal likelihoods:", log_marginal_likelihoods)
    print("Estimate of Theta:", Theta)
    print("Posterior mean of beta:", E_beta)

    plot_marginal_likelihood(log_marginal_likelihoods, 'outputs/log_marginal_likelihoods.png')

    # ---------------- LASSO -----------------
    lasso_params = {'alpha': 0.0005, 'fit_intercept': False, 'l1_ratio': 1.0, 'max_iter': 1000}
    lasso = ElasticNet(**lasso_params)
    
    lasso.fit(y, Z, X)
    y_pred = lasso.predict(Z, X)

    print('Train MSE:', mean_squared_error(y, y_pred)) # Calculate MSE
    
    omega_beta = lasso.coef_
    omega = omega_beta[:Z.shape[1]]
    beta = omega_beta[Z.shape[1]:]

    print("Estimated omega:", omega)
    print("Estimated beta:", beta)


    # ------------------- Search for the best lambda -------------------
    min_mse, min_average_row = lambda_search(y, Z, X, ElasticNet, lasso_params, lambda_list = np.arange(0.003, 0.01 + 0.0001, 0.0001), save_path='outputs/cv_mses_all.npy', cv_folds=10)
    print("Minimum MSE:", min_mse)
    print("Minimum average row:", min_average_row)