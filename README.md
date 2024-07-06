<!-- Consider a dataset $\{\boldsymbol{y}, \boldsymbol{Z}, \boldsymbol{X}\}$ with $n$ samples, where $\boldsymbol{y} \in \mathbb{R}^n$ is the vector of response variable, $\boldsymbol{Z} \in \mathbb{R}^{n\times c}$ is the matrix of $c$ independent variables, and $\boldsymbol{X} \in \mathbb{R}^{n\times p}$ is another matrix of $p$ variables. The linear mixed model (LMM) builds upon a linear relationship from $\boldsymbol{y}$ to $\{\boldsymbol{Z}, \boldsymbol{X}\}$ by
$$\boldsymbol{y} = \boldsymbol{Z}\boldsymbol{\omega} + \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{e},$$


where $\boldsymbol{y} \in \mathbb{R}^c$ is the vector of fixed effects, $\boldsymbol{\beta} \in \mathbb{R}^p$ is the vector of random effects with $\boldsymbol{\beta} \sim \mathbb{N}(0, \sigma^2_\beta \boldsymbol{I}_p)$, and $\boldsymbol{e} \sim \mathbb{N}(0, \sigma^2_e \boldsymbol{I})$ is the independent noise term. Let $\boldsymbol{\Theta}$ denote
the set of unknown parameters $\boldsymbol{\Theta} = \{\boldsymbol{\omega, \sigma^2_e, \sigma^2_\beta}\}$. We can treat $\boldsymbol{\beta}$ as a latent variable
because it is unobserved.

(a) Derive and implement an EM algorithm for the above linear mixed model and return the marginal likelihood at each iteration, the estimate of $\boldsymbol{\Theta}$, and the posterior mean of $\boldsymbol{\beta}$.

(b) Apply your implementation to the dataset stored here. The `lmm_y_z_x.txt` file has the first colum as $\boldsymbol{y}$. The columns labeled with Z.1 ∼ Z.10 correpspond to the matrix $\boldsymbol{Z}$ and the columns labeled with X.1 ∼ X.5000 correpspond to the matrix $\boldsymbol{X}$. Report the parameter estimates, visualize the marginal likelihood at each step of iteration and check the monotonicity.

(c) Let $\hat{\boldsymbol{\omega}}$ be the estimate of $\boldsymbol{\omega}$ and $\boldsymbol{\mu}$ be the posterior mean of $\boldsymbol{\beta}$. You can obtain
prediction $\hat{\boldsymbol{y}} = \boldsymbol{z}^T\hat{\boldsymbol{w}} + \boldsymbol{x}^T\boldsymbol{\mu}$, where $\boldsymbol{z}$ and $\boldsymbol{x}$ are from the testing dataset. Use cross-validation to evaluate your algorithm accuracy based on the dataset given in subproblem 2(b), and compare your method with the Lasso [Tibshirani, 1996]. You are required to implement the coordinate descent algorithm for Lasso by yourself. Please use the glmnet package in R Friedman et al. [2010], or sklearn.linear model in python as reference. The Lasso problem here should be
$$\frac{1}{2n}\|\boldsymbol{y} - \boldsymbol{Z}\boldsymbol{\omega} - \boldsymbol{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1,$$

(d) Suppose that we are using mean-field variational inference (MFVI) $q(\boldsymbol{\beta})$ to approximate the true posterior distribution $Pr(\boldsymbol{\beta}|\boldsymbol{X}, \boldsymbol{y};\boldsymbol{\Theta})$, where $q(\boldsymbol{\beta}) = \prod_{j=1}^p q(\beta_j)$. Derive an algorithm to obtain optimal mean-field approximation $q^*(\boldsymbol{\beta})$ and estimate model parameters $\boldsymbol{\Theta}$ (hint: in the E-step, you optimize $q(\boldsymbol{\beta})$, and in the M-step, you optimize $\boldsymbol{\Theta}$). Track both the marginal log-likelihood and the evidence lower bound (ELBO). Note the gap between them. Compare the posterior mean and variance of $\boldsymbol{\beta}$ obtained through MFVI and EM.




# Solution

To derive an EM algorithm for the linear mixed model (LMM), we need to find the Expectation (E) step and the Maximization (M) step. 

E-step:
We want to find the expectation of the log-likelihood function with respect to the conditional distribution of the latent variable $\boldsymbol{\beta}$ given the observed data $\boldsymbol{y}$ and the current estimate of the parameters $\boldsymbol{\Theta}^{(t)}$. The log-likelihood function is given by:

$$
\log L(\boldsymbol{\Theta}; \boldsymbol{y}) = \log p(\boldsymbol{y} \mid \boldsymbol{\Theta}) = \log \int p(\boldsymbol{y} \mid \boldsymbol{\beta}, \boldsymbol{\Theta}) p(\boldsymbol{\beta} \mid \boldsymbol{\Theta}) d\boldsymbol{\beta}
$$

The expectation in the E-step is:

$$
Q(\boldsymbol{\Theta} \mid \boldsymbol{\Theta}^{(t)}) = E_{\boldsymbol{\beta} \mid \boldsymbol{y}, \boldsymbol{\Theta}^{(t)}} [\log L(\boldsymbol{\Theta}; \boldsymbol{y})] = E_{\boldsymbol{\beta} \mid \boldsymbol{y}, \boldsymbol{\Theta}^{(t)}} [\log p(\boldsymbol{y} \mid \boldsymbol{\beta}, \boldsymbol{\Theta}) + \log p(\boldsymbol{\beta} \mid \boldsymbol{\Theta})]
$$

M-step:
We want to find the parameters $\boldsymbol{\Theta}$ that maximize the expectation found in the E-step:

$$
\boldsymbol{\Theta}^{(t+1)} = \text{argmax}_{\boldsymbol{\Theta}} Q(\boldsymbol{\Theta} \mid \boldsymbol{\Theta}^{(t)})
$$ -->