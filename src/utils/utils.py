import os
import torch
import numpy as np

CKPT_FTEMP = 'ckpt-ep-%d.pt'

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

def list_to_pth(list_of_arr, pth_dtype=torch.FloatTensor):
    return [to_pth(arr, pth_dtype=pth_dtype) for arr in list_of_arr]

def list_to_sqpth(list_of_arr, pth_dtype=torch.FloatTensor):
    return [to_sqpth(arr, pth_dtype=pth_dtype) for arr in list_of_arr]

def to_np(torch_tensor):
    return torch_tensor.data.numpy()

def to_sqnp(torch_tensor, dtype=np.float):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)

def init_ll(m, n):
    return [[[] for _ in range(n)] for _ in range(m)]

def init_lll(m, n, k):
    return [init_ll(n, k) for _ in range(m)]


def save_ckpt(cur_epoch, log_path, agent, optimizer, verbose=False):
    # compute fname
    ckpt_fname = CKPT_FTEMP % cur_epoch
    log_fpath = os.path.join(log_path, ckpt_fname)
    torch.save({
        'network_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, log_fpath)
    if verbose:
        print(f'model saved at epoch {cur_epoch}')



def load_ckpt(epoch_load, log_path, agent, optimizer=None):
    # compute fname
    ckpt_fname = CKPT_FTEMP % epoch_load
    log_fpath = os.path.join(log_path, ckpt_fname)
    if os.path.exists(log_fpath):
        # load the ckpt back
        checkpoint = torch.load(log_fpath)
        # unpack results
        agent.load_state_dict(checkpoint['network_state_dict'])
        if optimizer is None:
            optimizer = torch.optim.Adam(agent.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train()
        # msg
        print(f'network weights - epoch {epoch_load} loaded')
        return agent, optimizer
    print('ERROR: ckpt DNE')
    return None, None

if __name__ == "__main__":
    lll = init_lll(4,3,2)
    print(np.shape(lll))
