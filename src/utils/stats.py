import torch
import numpy as np
from scipy.stats import sem

def compute_stats(arr, axis=0, n_se=2, use_se=True):
    """compute mean and errorbar w.r.t to SE
    Parameters
    ----------
    arr : nd array
        data
    axis : int
        the axis to do stats along with
    n_se : int
        number of SEs
    Returns
    -------
    (n-1)d array, (n-1)d array
        mean and se
    """
    mu_ = np.mean(arr, axis=axis)
    if use_se:
        er_ = sem(arr, axis=axis) * n_se
    else:
        er_ = np.std(arr, axis=axis)
    return mu_, er_


def moving_average(x, winsize):
    return np.convolve(x, np.ones(winsize), 'valid') / winsize


def entropy(probs):
    """calculate entropy.
    I'm using log base 2!
    Parameters
    ----------
    probs : a torch vector
        a prob distribution
    Returns
    -------
    torch scalar
        the entropy of the distribution
    """
    return - torch.stack([pi * torch.log2(pi) for pi in probs]).sum()
