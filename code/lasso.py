import numbers
import sys
import warnings
import numpy as np
from scipy import sparse
from .cross_validation import cross_validate
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    check_random_state,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.linear_model._base import LinearModel, _check_precomputed_gram_matrix, _preprocess_data
from numbers import Integral, Real
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot


RAND_R_MAX = 2147483647


def enet_coordinate_descent(
    w,
    alpha,
    beta,
    X,
    y,
    max_iter,
    tol,
    rng = np.random.default_rng(),
    random=0,
    positive=0,
    b_index=10
):
    """Python version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """

    # get the data information into easy vars
    n_samples, n_features = X.shape

    # compute norms of the columns of X
    norm_cols_X = np.linalg.norm(X, axis=0) ** 2

    # initial value of the residuals
    R = y - X @ w
    tol *= np.dot(y, y)

    gap = tol + 1.0
    d_w_tol = tol
    n_iter = 0
    

    if alpha == 0 and beta == 0:
        warnings.warn("Coordinate descent with no regularization may lead to "
                      "unexpected results and is discouraged.")

    while n_iter < max_iter and gap >= tol:
        w_max = 0.0
        d_w_max = 0.0
        for ii in range(n_features):  # Loop over coordinates
            if norm_cols_X[ii] == 0.0:
                continue

            w_ii = w[ii]  # Store previous value

            if w_ii != 0.0:
                R += w_ii * X[:,ii]

            tmp = (X[:,ii] * R).sum()

            if positive and tmp < 0:
                w[ii] = 0.0
            else:
                w[ii] = (np.sign(tmp) * np.maximum(np.abs(tmp) - alpha, 0)
                            / (norm_cols_X[ii] + beta))

            if w[ii] != 0.0:
                R -=  w[ii] * X[:,ii] # Update residual

            # update the maximum absolute coefficient update
            d_w_ii = np.abs(w[ii] - w_ii)
            d_w_max = max(d_w_max, d_w_ii)

            w_max = max(w_max, np.abs(w[ii]))

        if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
            # the biggest coordinate update of this iteration was smaller
            # than the tolerance: check the duality gap as ultimate
            # stopping criterion

            XtA = np.dot(X.T, R) - beta * w

            if positive:
                dual_norm_XtA = np.max(np.abs(XtA))
            else:
                dual_norm_XtA = np.linalg.norm(XtA, ord=np.inf)

            R_norm2 = np.dot(R, R)

            if dual_norm_XtA > alpha:
                const = alpha / dual_norm_XtA
                A_norm2 = R_norm2 * (const ** 2)
                gap = 0.5 * (R_norm2 + A_norm2) / n_samples
            else:
                const = 1.0
                gap = R_norm2

            l1_norm = np.sum(np.abs(w[b_index:]))

            gap += l1_norm - const * np.dot(R, y)
        
        n_iter += 1


    if n_iter == max_iter and gap >= tol:
        warnings.warn("Objective did not converge. Consider increasing iterations, feature scaling, or regularization.", stacklevel=2)

    return np.asarray(w), gap, tol, n_iter + 1


def _pre_fit(
    X,
    y,
    Xy,
    precompute,
    normalize,
    fit_intercept,
    copy,
    check_input=True,
    sample_weight=None,
):
    """Function used at beginning of fit in linear models with L1 or L0 penalty.

    This function applies _preprocess_data and additionally computes the gram matrix
    `precompute` as needed as well as `Xy`.
    """
    n_samples, n_features = X.shape

    # copy was done in fit if necessary
    X, y, X_offset, y_offset, X_scale = _preprocess_data(
        X,
        y,
        fit_intercept=fit_intercept,
        normalize=normalize,
        copy=copy,
        check_input=check_input,
        sample_weight=sample_weight,
    )
    # Rescale only in dense case. Sparse cd solver directly deals with
    # sample_weight.

    # FIXME: 'normalize' to be removed in 1.4
    if hasattr(precompute, "__array__"):
        if (
            fit_intercept
            and not np.allclose(X_offset, np.zeros(n_features))
            or normalize
            and not np.allclose(X_scale, np.ones(n_features))
        ):
            warnings.warn(
                "Gram matrix was provided but X was centered to fit "
                "intercept, or X was normalized : recomputing Gram matrix.",
                UserWarning,
            )
            # recompute Gram
            precompute = "auto"
            Xy = None
        elif check_input:
            # If we're going to use the user's precomputed gram matrix, we
            # do a quick check to make sure its not totally bogus.
            _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale)

    # precompute if n_samples > n_features
    if isinstance(precompute, str) and precompute == "auto":
        precompute = n_samples > n_features

    if precompute is True:
        # make sure that the 'precompute' array is contiguous.
        precompute = np.empty(shape=(n_features, n_features), dtype=X.dtype, order="C")
        np.dot(X.T, X, out=precompute)

    if not hasattr(precompute, "__array__"):
        Xy = None  # cannot use Xy if precompute is not Gram

    if hasattr(precompute, "__array__") and Xy is None:
        common_dtype = np.result_type(X.dtype, y.dtype)
        if y.ndim == 1:
            # Xy is 1d, make sure it is contiguous.
            Xy = np.empty(shape=n_features, dtype=common_dtype, order="C")
            np.dot(X.T, y, out=Xy)
        else:
            # Make sure that Xy is always F contiguous even if X or y are not
            # contiguous: the goal is to make it fast to extract the data for a
            # specific target.
            n_targets = y.shape[1]
            Xy = np.empty(shape=(n_features, n_targets), dtype=common_dtype, order="F")
            np.dot(y.T, X, out=Xy.T)

    return X, y, X_offset, y_offset, X_scale, precompute, Xy


def _set_order(X, y, order="C"):
    """Change the order of X and y if necessary.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Target values.

    order : {None, 'C', 'F'}
        If 'C', dense arrays are returned as C-ordered, sparse matrices in csr
        format. If 'F', dense arrays are return as F-ordered, sparse matrices
        in csc format.

    Returns
    -------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data with guaranteed order.

    y : ndarray of shape (n_samples,)
        Target values with guaranteed order.
    """
    if order not in [None, "C", "F"]:
        raise ValueError(
            "Unknown value for order. Got {} instead of None, 'C' or 'F'.".format(order)
        )
    sparse_X = sparse.issparse(X)
    sparse_y = sparse.issparse(y)
    if order is not None:
        sparse_format = "csc" if order == "F" else "csr"
        if sparse_X:
            X = X.asformat(sparse_format, copy=False)
        else:
            X = np.asarray(X, order=order)
        if sparse_y:
            y = y.asformat(sparse_format)
        else:
            y = np.asarray(y, order=order)
    return X, y


def enet_path(
    X,
    y,
    *,
    l1_ratio=0.5,
    eps=1e-3,
    n_alphas=100,
    alphas=None,
    precompute="auto",
    Xy=None,
    copy_X=True,
    coef_init=None,
    verbose=False,
    return_n_iter=False,
    positive=False,
    check_input=True,
    **params,
):
    """Compute elastic net path with coordinate descent.

    The elastic net optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : {array-like, sparse matrix} of shape (n_samples,) or \
        (n_samples, n_targets)
        Target values.

    l1_ratio : float, default=0.5
        Number between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If None alphas are set automatically.

    precompute : 'auto', bool or array-like of shape \
            (n_features, n_features), default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like of shape (n_features,) or (n_features, n_targets),\
         default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : ndarray of shape (n_features, ), default=None
        The initial values of the coefficients.

    verbose : bool or int, default=False
        Amount of verbosity.

    return_n_iter : bool, default=False
        Whether to return the number of iterations or not.

    positive : bool, default=False
        If set to True, forces coefficients to be positive.
        (Only allowed when ``y.ndim == 1``).

    check_input : bool, default=True
        If set to False, the input validation checks are skipped (including the
        Gram matrix when provided). It is assumed that they are handled
        by the caller.

    **params : kwargs
        Keyword arguments passed to the coordinate descent solver.

    Returns
    -------
    alphas : ndarray of shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : ndarray of shape (n_features, n_alphas) or \
            (n_targets, n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : ndarray of shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : list of int
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).

    See Also
    --------
    MultiTaskElasticNet : Multi-task ElasticNet model trained with L1/L2 mixed-norm \
    as regularizer.
    MultiTaskElasticNetCV : Multi-task L1/L2 ElasticNet with built-in cross-validation.
    ElasticNet : Linear regression with combined L1 and L2 priors as regularizer.
    ElasticNetCV : Elastic Net model with iterative fitting along a regularization path.

    Notes
    -----
    For an example, see
    :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.
    """
    X_offset_param = params.pop("X_offset", None)
    X_scale_param = params.pop("X_scale", None)
    sample_weight = params.pop("sample_weight", None)
    tol = params.pop("tol", 1e-4)
    max_iter = params.pop("max_iter", 1000)
    random_state = params.pop("random_state", None)
    selection = params.pop("selection", "cyclic")

    if len(params) > 0:
        raise ValueError("Unexpected parameters in params", params.keys())

    # We expect X and y to be already Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(
            X,
            accept_sparse="csc",
            dtype=[np.float64, np.float32],
            order="F",
            copy=copy_X,
        )
        y = check_array(
            y,
            accept_sparse="csc",
            dtype=X.dtype.type,
            order="F",
            copy=False,
            ensure_2d=False,
        )
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(
                Xy, dtype=X.dtype.type, order="C", copy=False, ensure_2d=False
            )

    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        n_targets = y.shape[1]

    if multi_output and positive:
        raise ValueError("positive=True is not allowed for multi-output (y.ndim != 1)")

    # MultiTaskElasticNet does not support sparse matrices
    if not multi_output and sparse.isspmatrix(X):
        if X_offset_param is not None:
            # As sparse matrices are not actually centered we need this to be passed to
            # the CD solver.
            X_sparse_scaling = X_offset_param / X_scale_param
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    n_alphas = len(alphas)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(random_state)
    if selection not in ["random", "cyclic"]:
        raise ValueError("selection should be either random or cyclic.")
    random = selection == "random"

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_targets, n_features, n_alphas), dtype=X.dtype)

    if coef_init is None:
        coef_ = np.zeros(coefs.shape[:-1], dtype=X.dtype, order="F")
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)

    for i, alpha in enumerate(alphas):
        # account for n_samples scaling in objectives between here and cd_fast
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples

        model = enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
            )
        
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        # we correct the scale of the returned dual gap, as the objective
        # in cd_fast is n_samples * the objective in this docstring.
        dual_gaps[i] = dual_gap_ / n_samples
        n_iters.append(n_iter_)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print("Path: %03i out of %03i" % (i, n_alphas))
            else:
                sys.stderr.write(".")

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps


class ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel):
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * ||w||_1 + 0.5 * b * ||w||_2^2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter. ``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool, default=True
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.

    max_iter : int, default=1000
        The maximum number of iterations.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``, see Notes below.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).

    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_targets, n_features)
        Sparse representation of the `coef_`.

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : list of int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : float or ndarray of shape (n_targets,)
        Given param alpha, the dual gaps at the end of the optimization,
        same shape as each observation of y.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ElasticNetCV : Elastic net model with best model selection by
        cross-validation.
    SGDRegressor : Implements elastic net regression with incremental training.
    SGDClassifier : Implements logistic regression with elastic net penalty
        (``SGDClassifier(loss="log_loss", penalty="elasticnet")``).

    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    The precise stopping criteria based on `tol` are the following: First, check that
    that maximum coordinate update, i.e. :math:`\\max_j |w_j^{new} - w_j^{old}|`
    is smaller than `tol` times the maximum absolute coefficient, :math:`\\max_j |w_j|`.
    If so, then additionally check whether the dual gap is smaller than `tol` times
    :math:`||y||_2^2 / n_{\text{samples}}`.

    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "l1_ratio": [Interval(Real, 0, 1, closed="both")],
        "fit_intercept": ["boolean"],
        "precompute": ["boolean", "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "copy_X": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "warm_start": ["boolean"],
        "positive": ["boolean"],
        "random_state": ["random_state"],
        "selection": [StrOptions({"cyclic", "random"})],
    }

    path = staticmethod(enet_path)

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, y_ori, Z_ori, X_ori, sample_weight=None, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of (n_samples, n_features)
            Data.

        y : {ndarray, sparse matrix} of shape (n_samples,) or \
            (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.

        sample_weight : float or array-like of shape (n_samples,), default=None
            Sample weights. Internally, the `sample_weight` vector will be
            rescaled to sum to `n_samples`.

            .. versionadded:: 0.23

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        X = np.hstack((Z_ori, X_ori))
        y = y_ori

        self._validate_params()

        if self.alpha == 0:
            warnings.warn(
                "With alpha=0, this algorithm does not converge "
                "well. You are advised to use the LinearRegression "
                "estimator",
                stacklevel=2,
            )

        # Remember if X is copied
        X_copied = False

        n_samples, n_features = X.shape
        alpha = self.alpha

        if isinstance(sample_weight, numbers.Number):
            sample_weight = None
        if sample_weight is not None:
            if check_input:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
            # TLDR: Rescale sw to sum up to n_samples.
            # Long: The objective function of Enet
            #
            #    1/2 * np.average(squared error, weights=sw)
            #    + alpha * penalty                                             (1)
            #
            # is invariant under rescaling of sw.
            # But enet_path coordinate descent minimizes
            #
            #     1/2 * sum(squared error) + alpha' * penalty                  (2)
            #
            # and therefore sets
            #
            #     alpha' = n_samples * alpha                                   (3)
            #
            # inside its function body, which results in objective (2) being
            # equivalent to (1) in case of no sw.
            # With sw, however, enet_path should set
            #
            #     alpha' = sum(sw) * alpha                                     (4)
            #
            # Therefore, we use the freedom of Eq. (1) to rescale sw before
            # calling enet_path, i.e.
            #
            #     sw *= n_samples / sum(sw)
            #
            # such that sum(sw) = n_samples. This way, (3) and (4) are the same.
            sample_weight = sample_weight * (n_samples / np.sum(sample_weight))
            # Note: Alternatively, we could also have rescaled alpha instead
            # of sample_weight:
            #
            #     alpha *= np.sum(sample_weight) / n_samples

        # Ensure copying happens only once, don't do it again if done above.
        # X and y will be rescaled if sample_weight is not None, order='F'
        # ensures that the returned X and y are still F-contiguous.
        should_copy = self.copy_X and not X_copied
        
        X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(
            X,
            y,
            None,
            self.precompute,
            normalize=False,
            fit_intercept=self.fit_intercept,
            copy=should_copy,
            check_input=check_input,
            sample_weight=sample_weight,
        )
        # coordinate descent needs F-ordered arrays and _pre_fit might have
        # called _rescale_data
        if check_input or sample_weight is not None:
            X, y = _set_order(X, y, order="F")
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_targets = y.shape[1]

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype, order="F")
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []

        for k in range(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            _, this_coef, this_dual_gap, this_iter = self.path(
                X,
                y[:, k],
                l1_ratio=self.l1_ratio,
                eps=None,
                n_alphas=None,
                alphas=[alpha],
                precompute=precompute,
                Xy=this_Xy,
                copy_X=True,
                coef_init=coef_[k],
                verbose=False,
                return_n_iter=True,
                positive=self.positive,
                check_input=False,
                # from here on **params
                tol=self.tol,
                X_offset=X_offset,
                X_scale=X_scale,
                max_iter=self.max_iter,
                random_state=self.random_state,
                selection=self.selection,
                sample_weight=sample_weight,
            )
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]
            self.coef_ = coef_[0]
            self.dual_gap_ = dual_gaps_[0]
        else:
            self.coef_ = coef_
            self.dual_gap_ = dual_gaps_

        self._set_intercept(X_offset, y_offset, X_scale)

        # check for finiteness of coefficients
        if not all(np.isfinite(w).all() for w in [self.coef_, self.intercept_]):
            raise ValueError(
                "Coordinate descent iterations resulted in non-finite parameter"
                " values. The input data may contain large values and need to"
                " be preprocessed."
            )

        # return self for chaining fit and predict calls
        return self
    
    def predict(self, Z_ori, X_ori):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        X = np.hstack((Z_ori, X_ori))
        return self._decision_function(X)

    @property
    def sparse_coef_(self):
        """Sparse representation of the fitted `coef_`."""
        return sparse.csr_matrix(self.coef_)

    def _decision_function(self, X):
        """Decision function of the linear model.

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : ndarray of shape (n_samples,)
            The predicted decision function.
        """
        check_is_fitted(self)
        if sparse.isspmatrix(X):
            return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        else:
            return super()._decision_function(X)




def lambda_search(y, Z, X, model, lasso_params, lambda_list = np.arange(0.0001, 0.001 + 0.0001, 0.0001), save_path='cv_mses_all.npy', cv_folds=10):
    """
    Perform cross-validation to find the best lambda for the LASSO model
    """

    cv_mses_all = np.zeros((len(lambda_list), cv_folds + 1))
    cv_mses_all[:, 0] = lambda_list
   
    for i, l in enumerate(lambda_list):
        lasso_params['alpha'] = l
        cv_mses = cross_validate(y, Z, X, model, lasso_params, n_splits=cv_folds)
        cv_mses_all[i, 1:] = cv_mses

    np.save(save_path, cv_mses_all)

    average_values = np.mean(cv_mses_all[:, 1:], axis=1)
    min_average_index = np.argmin(average_values)
    min_average_row = cv_mses_all[min_average_index]
    min_mse = average_values[min_average_index]

    return min_mse, min_average_row