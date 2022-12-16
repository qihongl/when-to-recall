import os
import glob
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


def get_recall_info(cache):
    # full ver
    [vector_signal, scalar_signal, misc] = cache
    # [h_t, m_t, cm_t, dec_act_t, mems, ma_t] = misc
    # [f_t, i_t, o_t] = vector_signal
    # [inps, _, _]= scalar_signal
    [_, _, _, _, _, ma_t] = misc
    [emg_t, _, _]= scalar_signal
    return emg_t, ma_t

def list_fnames(data_dir, fpattern):
    '''
    list all fnames/fpaths with a particular fpattern (e.g. *pca.pkl)
    '''
    fpaths = glob.glob(os.path.join(data_dir, fpattern))
    n_data_files = len(fpaths)
    fnames = [None] * n_data_files
    for i, fpath in enumerate(fpaths):
        # get file info
        fnames[i] = os.path.basename(fpath)
    return fpaths, fnames

def ckpt_exists(log_dir):
    _, fnames = list_fnames(log_dir, 'ckpt*.pt')
    if len(fnames) > 0 :
        return True
    return False

def get_max_epoch_saved(log_dir):
    _, fnames = list_fnames(log_dir, 'ckpt*.pt')
    epoch_saved = [int(fname.split('-')[-1].split('.')[0]) for fname in fnames]
    return max(epoch_saved)


if __name__ == "__main__":
    lll = init_lll(4,3,2)
    print(np.shape(lll))
