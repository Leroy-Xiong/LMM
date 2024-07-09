from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.stats import multivariate_normal


class EMAlgorithm:
    def __init__(self, max_iter=200, tol=1e-6, omega=np.zeros(10), sigma_b2=1, sigma_e2=1):
        self.max_iter = max_iter
        self.tol = tol
        self.omega_init = omega
        self.sigma_b2_init = sigma_b2
        self.sigma_e2_init = sigma_e2
    

    def fit(self, y, Z, X):
        """
        EM algorithm for estimating the parameters of linear mixed model with Gaussian noise
        
        Parameters
        y : array-like, shape (n,)
            The response variable
        Z : array-like, shape (n, c)
            The covariates
        X : array-like, shape (n, p)
            The covariates
        max_iter : int, optional
            The maximum number of iterations, by default 200
        tol : float, optional
            The tolerance for the stopping criterion, by default 1e-6
        omega : array-like, shape (c,), optional
            initial value of omega, by default np.random.randn(c)
        sigma_b2 : float, optional
            initial value of sigma_b2, by default 1
        sigma_e2 : float, optional
            initial value of sigma_e2, by default 1

        Returns
        log_marginal_likelihoods : array-like, shape (max_iter,)
            log marginal likelihoods
        Theta : array-like, shape (c + 2,)
            Theta = (omega, sigma_b2, sigma_e2)
        E_beta : array-like, shape (p,)
            posterior mean of beta
        """

        n, c = Z.shape
        _, p = X.shape
        
        # Initialize parameters
        lm_fit = LinearRegression(fit_intercept=False).fit(Z, y)
        omega = lm_fit.coef_
        predictions = lm_fit.predict(Z)
        residuals = y - predictions
        sigma_e2 = np.sum(residuals ** 2) / (n - c)

        sigma_b2 = self.sigma_b2_init
        
        log_marginal_likelihoods = []

        XXT = X @ X.T
        XXT_eigen_values, XXT_eigen_vectors = np.linalg.eig(XXT)
        
        for i in tqdm(range(self.max_iter), desc='EM Algorithm'): 

            # E-step
            Diag = XXT_eigen_values / sigma_e2 + 1 / sigma_b2

            mu = 1 / sigma_e2 * X.T @ (XXT_eigen_vectors @ (XXT_eigen_vectors.T @ (y - Z @ omega) / Diag))

            SigmaXTX_tr = np.sum(1 / (1 + sigma_e2 / sigma_b2 / XXT_eigen_values)) * sigma_e2

            Sigma_tr = (np.sum(1 / (XXT_eigen_values + sigma_e2 / sigma_b2)) + (p - n) / (sigma_e2 / sigma_b2)) * sigma_e2

            muTmu = mu.T @ mu
            Xmu = X @ mu

            # Compute marginal likelihood
            log_marginal_likelihood = -np.sum(np.log(Diag)) / 2 - n / 2 * np.log(sigma_b2) - n / 2 * np.log(sigma_e2) - n / 2 * np.log(2 * np.pi) - 1 / 2 * np.sum((y - Z @ omega) ** 2 / Diag)

            # Check for convergence
            if len(log_marginal_likelihoods) > 0 and np.abs(log_marginal_likelihood - log_marginal_likelihoods[-1]) < self.tol:
                break

            log_marginal_likelihoods.append(log_marginal_likelihood)
            
            # M-step
            omega = np.linalg.solve(Z.T @ Z, Z.T) @ (y - X @ mu)
            sigma_e2 = ((y - Z @ omega - Xmu).T @ (y - Z @ omega - Xmu) + SigmaXTX_tr) / n
            sigma_b2 = (muTmu + Sigma_tr) / p   

        

        self.n = n
        self.p = p
        self.c = c
        self.log_marginal_likelihoods = log_marginal_likelihoods
        self.Theta = np.hstack((omega, sigma_b2, sigma_e2))
        self.E_beta = mu
        return log_marginal_likelihoods, self.Theta, self.E_beta

    def predict(self, Z, X):
        omega = self.Theta[:self.c]
        E_beta = self.E_beta

        y_hat = Z @ omega + X @ E_beta

        return y_hat

    def plot_marginal_likelihood(self, save_path):
        plt.figure(figsize=(5, 3), dpi=300)
        plt.plot(self.log_marginal_likelihoods)
        plt.xlabel('Iteration')
        plt.ylabel('Log Marginal Likelihood')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()