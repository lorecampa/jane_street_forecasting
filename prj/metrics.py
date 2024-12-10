import numpy as np

def weighted_r2(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    return 1 - (np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights * (y_true ** 2)))

def weighted_mae(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def weighted_mse(y_true, y_pred, weights):
    if weights is None:
        weights = np.ones_like(y_true)
    return np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights)

def weighted_rmse(y_true, y_pred, weights):
    if weights is None:
        weights = np.ones_like(y_true)
    mse = np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights)
    return np.sqrt(mse)


def squared_weighted_error_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w.reshape(-1, 1) * ((y_true.reshape(-1, 1) - y_pred) ** 2)

def absolute_weighted_error_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w.reshape(-1, 1) * np.abs(y_true.reshape(-1, 1) - y_pred)

def log_cosh_weighted_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:    
    return w.reshape(-1, 1) * np.log(np.cosh(y_true.reshape(-1, 1) - y_pred))

def r2_weighted_loss_fn(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1 - weighted_r2(y_true, y_pred, weights=w)