import numpy as np
import empyrical as ep


def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    return ep.cum_returns(returns_pct)


def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return float
    """
    returns = np.array(returns_pct)
    if returns.std() == 0:
        sharpe_ratio = np.inf
    else:
        sharpe_ratio = (returns.mean() - risk_free) / returns.std()
    return sharpe_ratio


def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    return ep.max_drawdown(returns_pct)


def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct)[len(returns_pct) - 1]
    if mdd == 0:
        return np.inf
    return returns / mdd


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