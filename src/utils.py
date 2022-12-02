import torch
import numpy as np

def onehot(n, k):
    '''
    return a n-D one hot vector and the k-th entry is 1
    * k uses 0 based indexing, so onehot(n, n) is invalid
    '''
    assert k <= n
    return np.eye(n)[k]

def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.tensor(np_array).type(pth_dtype)


def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))


def to_np(torch_tensor):
    return torch_tensor.data.numpy()


def to_sqnp(torch_tensor, dtype=np.float):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)
