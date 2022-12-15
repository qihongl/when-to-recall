import os

LOG_ROOT = '../log'
GATING_TYPES = ['pre', 'post']

class Parameters():

    def __init__(
        self,
        B = 10,
        penalty = 10,
        n_hidden = 128,
        lr = 1e-3,
        cmpt = .5,
        eta = 0.1,
        test_mode = True,
        add_query_indicator = True,
        gating_type = 'pre',
        n_epochs = 4000,
        sup_epoch = 2000
    ):
        assert B >= 2
        assert penalty >= 0
        assert n_hidden >= 1
        assert lr > 0
        assert eta > 0
        assert cmpt > 0
        assert type(add_query_indicator) == type(True)
        assert type(test_mode) == type(True)
        assert gating_type in GATING_TYPES
        assert n_epochs >= sup_epoch >= 0
        self.B = B
        self.penalty = penalty
        self.n_hidden = n_hidden
        self.lr = lr
        self.cmpt = cmpt
        self.eta = eta
        self.test_mode = test_mode
        self.add_query_indicator = add_query_indicator
        self.gating_type = gating_type
        self.n_epochs = n_epochs
        self.sup_epoch = sup_epoch
        self.log_dirs = f'B-{B}/n_hidden-{n_hidden}-lr-{lr}/gating-{gating_type}/task-unit-{int(add_query_indicator)}/penalty-{penalty}/train-mode-{int(test_mode)}-n_epochs-{n_epochs}-sup_epoch-{sup_epoch}/'

    def gen_log_dirs(self, verbose=False):
        fpath = os.path.join(LOG_ROOT, self.log_dirs)
        mkdir_ifdne(fpath, verbose)
        return fpath

def mkdir_ifdne(dir_name, verbose=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if verbose:
            print(f'Dir exist: {dir_name}')

if __name__ == "__main__":
    B = 10
    penalty = 10
    n_hidden = 128
    lr = 1e-3
    cmpt = .5
    eta = 0.1
    test_mode = True
    add_query_indicator = True
    gating_type = 'pre'
    n_epochs = 4000
    sup_epoch = 2000

    p = Parameters(
        B = B,
        penalty = penalty,
        n_hidden = n_hidden,
        lr = lr,
        cmpt = cmpt,
        eta = eta,
        test_mode = test_mode,
        add_query_indicator = add_query_indicator,
        gating_type = gating_type,
        n_epochs = n_epochs,
        sup_epoch = sup_epoch
    )

    fpath = p.gen_log_dirs(verbose=True)
    print(fpath)
