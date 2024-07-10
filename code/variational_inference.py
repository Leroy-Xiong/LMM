from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


class VariationalInference:
    def __init__(self, max_iter=200, tol=1e-6, omega=np.zeros(10), sigma_b2=1, sigma_e2=1):
        self.max_iter = max_iter
        self.tol = tol
        self.omega_init = omega
        self.sigma_b2_init = sigma_b2
        self.sigma_e2_init = sigma_e2
        self.n = None
        self.p = None
        self.c = None
        self.log_marginal_likelihoods = None
        self.ELBOs = None
        self.Theta = None
        self.m = None

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
        Theta = np.hstack((omega, sigma_b2, sigma_e2))

        m = np.zeros(p)
        s2 = np.ones(p)
        
        log_marginal_likelihoods = []
        ELBOs = []

        ZTZZT = np.linalg.inv(Z.T @ Z) @ Z.T
        XXT = X @ X.T
        XTX = X.T @ X

        XXT_eigen_values, _ = np.linalg.eig(XXT)
        
        for _ in tqdm(range(self.max_iter)): 
            
            s2 = np.zeros(p)
            m = np.zeros(p)

            # E-step
            for j in range(p):
                s2[j] = 1 / (X[:, j].T @ X[:, j] / sigma_e2 + 1 / sigma_b2)
                m[j] = s2[j] / sigma_e2 * (y - Z @ omega).T @ X[:, j]
            
            # M-step
            omega_new = ZTZZT @ (y - X @ m)
            sigma_b2_new = (m.T @ m + np.sum(s2)) / p  
            sigma_e2_new = ((y - Z @ omega_new - X @ m).T @ (y - Z @ omega - X @ m) + np.trace(np.diag(s2) @ XTX)) / n

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

            print(Theta)
            print(np.mean(m))
            print(np.mean(s2))

            # Compute marginal likelihood
            # Diag = XXT_eigen_values / sigma_e2 + 1 / sigma_b2

            # log_marginal_likelihood = -np.sum(np.log(Diag)) / 2 - n / 2 * np.log(sigma_b2) - n / 2 * np.log(sigma_e2) - n / 2 * np.log(2 * np.pi) - 1 / 2 * np.sum((y - Z @ omega) ** 2 / Diag)


            # E_1 = -(n + p) / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma_e2) - p / 2 * np.log(sigma_b2) - 1 / (2 * sigma_e2) * (y - Z @ omega).T @ (y - Z @ omega) + 1 / sigma_e2 * (y - Z @ omega).T @ X @ m - 1 / (2 * sigma_e2) * (np.trace(X @ np.diag(s2) @ X.T) + m.T @ X.T @ X @ m) - 1 / (2 * sigma_b2) * (np.sum(s2) + m.T @ m)
            # E_2 = -p / 2 * np.log(2 * np.pi) - 1 / 2 * np.sum(np.log(s2)) - p / 2
            # ELBO = E_1 + E_2

            # log_marginal_likelihoods.append(log_marginal_likelihood)
            # ELBOs.append(ELBO)

            # print(log_marginal_likelihood)
            # print(ELBO)

        self.n = n
        self.p = p
        self.c = c
        self.log_marginal_likelihoods = np.array(log_marginal_likelihoods)
        self.ELBOs = np.array(ELBOs)
        self.Theta = Theta
        self.E_beta = m
        self.E_beta_2_diag = s2

        np.savetxt('outputs/log_marginal_likelihoods_vi.csv', log_marginal_likelihoods, delimiter=',')
        np.savetxt('outputs/ELBOs.csv', ELBOs, delimiter=',')

        np.save('outputs/E_beta_vi.npy', self.E_beta)
        np.save('outputs/E_beta_2_diag_vi.npy', self.E_beta_2_diag)

        return log_marginal_likelihoods, Theta

    def predict(self, Z, X):
        omega = self.Theta[:self.c]
        E_beta = self.E_beta

        y_hat = Z @ omega + X @ E_beta

        return y_hat

    def plot_marginal_likelihood_and_elbo(self, save_path):
        if self.log_marginal_likelihoods is None:
            self.log_marginal_likelihoods = np.genfromtxt('outputs/log_marginal_likelihoods_vi.csv')
        if self.ELBOs is None:
            self.ELBOs = np.genfromtxt('outputs/ELBOs.csv')

        plt.figure(figsize=(5, 4), dpi=300)
        plt.plot(self.log_marginal_likelihoods)
        plt.plot(self.ELBOs)
        plt.legend(['Evidence', 'ELBO'])
        plt.xlabel('Iteration')
        # plt.yscale('symlog')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_gap(self, save_path):
        """
        Plot the gap between the log marginal likelihood and the ELBO.
        :param save_path: The path to save the plot.
        :return: None
        """
        plt.figure(figsize=(5, 4), dpi=300)
        plt.plot(self.log_marginal_likelihoods - self.ELBOs, color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Gap')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()