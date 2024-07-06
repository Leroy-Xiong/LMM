import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def cross_validate(y, Z, X, model_class, model_params, n_splits=5):
    # seperate data into train and test
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    mses = np.zeros(n_splits)
    
    for i, (train_index, test_index) in enumerate(kf.split(y)):
        y_train, y_test = y[train_index], y[test_index]
        Z_train, Z_test = Z[train_index], Z[test_index]
        X_train, X_test = X[train_index], X[test_index]
        # fit model
        model = model_class(**model_params)
        model.fit(y_train, Z_train, X_train)
        # Predict test set
        y_pred = model.predict(Z_test, X_test)
        # Calculate MSE
        mses[i] = mean_squared_error(y_pred, y_test)
    
    return mses