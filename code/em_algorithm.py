from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


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
        omega = self.omega_init
        sigma_b2 = self.sigma_b2_init
        sigma_e2 = self.sigma_e2_init
        Theta = np.hstack((omega, sigma_b2, sigma_e2))
        
        log_marginal_likelihoods = []

        ZTZZT = np.linalg.inv(Z.T @ Z) @ Z.T
        XTX = X.T @ X
        
        for _ in tqdm(range(self.max_iter)): 

            # E-step
            Sigma = np.linalg.inv(XTX / sigma_e2 + np.eye(p) / sigma_b2)
            mu = Sigma @ X.T @ (y - Z @ omega) / sigma_e2
            E_beta = mu
            E_beta2 = Sigma + np.outer(mu.T, mu)
            
            # M-step
            omega_new = ZTZZT @ (y - X @ E_beta)
            sigma_b2_new = np.trace(E_beta2) / p
            sigma_e2_new = ((y - Z @ omega_new).T @ (y - Z @ omega_new) + np.trace(XTX @ E_beta2) - 2 * (y - Z @ omega_new).T @ X @ E_beta) / n
            
            # print(omega_new)
            # print(sigma_b2_new)
            # print(sigma_e2_new)

            # break
            Theta_new = np.hstack((omega_new, sigma_b2_new, sigma_e2_new))
        
            # Check for convergence
            if np.linalg.norm(Theta_new - Theta) < self.tol:
                break

            # Update parameters
            omega = omega_new
            sigma_e2 = sigma_e2_new
            sigma_b2 = sigma_b2_new
            Theta = Theta_new
            print(Theta_new)

            # Compute marginal likelihood
            log_marginal_likelihood = (-(n + p) / 2 - 2 / p) * np.log(2 * np.pi) - n / 2 * np.log(sigma_e2) - p / 2 * np.log(sigma_b2) - 1 / (2 * sigma_e2) * (y - Z @ omega).T @ (y - Z @ omega) + 1 / 2 * mu.T @ np.linalg.inv(Sigma) @ mu
            # log_marginal_likelihood = (-(n + p) / 2 - 2 / p) * np.log(2 * np.pi) - n / 2 * np.log(sigma_e2) - p / 2 * np.log(sigma_b2) - 2 * np.log(np.linalg.det(Sigma)) - 1 / (2 * sigma_e2) * (y - Z @ omega).T @ (y - Z @ omega) + 1 / 2 * mu.T @ np.linalg.inv(Sigma) @ mu
            print(log_marginal_likelihood)
            log_marginal_likelihoods.append(log_marginal_likelihood)

        self.n = n
        self.p = p
        self.c = c
        self.log_marginal_likelihoods = log_marginal_likelihoods
        self.Theta = Theta
        self.E_beta = E_beta
        return log_marginal_likelihoods, Theta, E_beta

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