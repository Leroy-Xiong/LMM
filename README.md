



# EM Algorithm and Variational Inference for Linear Mixed Model


Consider a dataset $\{\boldsymbol{y}, \boldsymbol{Z}, \boldsymbol{X}\}$ with $n$ samples, where $\boldsymbol{y} \in \mathbb{R}^n$ is the vector of response variable, $\boldsymbol{Z} \in \mathbb{R}^{n\times c}$ is the matrix of $c$ independent variables, and $\boldsymbol{X} \in \mathbb{R}^{n\times p}$ is another matrix of $p$ variables. The linear mixed model (LMM) builds upon a linear relationship from $\boldsymbol{y}$ to $\{\boldsymbol{Z}, \boldsymbol{X}\}$ by
$$\boldsymbol{y} = \boldsymbol{Z}\boldsymbol{\omega} + \boldsymbol{X}\boldsymbol{\beta} + \boldsymbol{e},$$


where $\boldsymbol{y} \in \mathbb{R}^c$ is the vector of fixed effects, $\boldsymbol{\beta} \in \mathbb{R}^p$ is the vector of random effects with $\boldsymbol{\beta} \sim \mathbb{N}(0, \sigma^2_\beta \boldsymbol{I}_p)$, and $\boldsymbol{e} \sim \mathbb{N}(0, \sigma^2_e \boldsymbol{I})$ is the independent noise term. Let $\boldsymbol{\Theta}$ denote
the set of unknown parameters $\boldsymbol{\Theta} = \{\boldsymbol{\omega, \sigma^2_e, \sigma^2_\beta}\}$. We can treat $\boldsymbol{\beta}$ as a latent variable
because it is unobserved.

This project implements EM algorithm and variational inference for the LMM. They both estimate the posterior distribution of $\boldsymbol{\Theta}$ given the observed data.