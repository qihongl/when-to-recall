import os

GATING_TYPES = ['pre', 'post']


class Parameters():

    def __init__(
        self,
        subj_id = 0,
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
        sup_epoch = 2000,
        verbose=True,
        exp_name='testing',
        log_root='../log',
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
        self.subj_id = subj_id
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
        self.exp_name = exp_name
        self.log_root = log_root
        sub_dirs = f'{exp_name}/B-{B}/n_hidden-{n_hidden}-lr-{lr}/gating-{gating_type}/task-unit-{int(add_query_indicator)}/cmpt-{cmpt}/eta-{eta}/penalty-{penalty}/train-mode-{int(test_mode)}-n_epochs-{n_epochs}-sup_epoch-{sup_epoch}/subj_id-{subj_id}/'
        # sub_dirs = f'{exp_name}/B-{B}/nH-{n_hidden}-lr-{lr}/{gating_type}/TU-{int(add_query_indicator)}/cmpt-{cmpt}/p-{penalty}/tr-{int(test_mode)}-ep-{n_epochs}/s-{subj_id}/'
        self.log_dir = os.path.join(log_root, sub_dirs)
        self.gen_log_dirs(verbose=verbose)

    def gen_log_dirs(self, verbose=False):
        mkdir_ifdne(self.log_dir, verbose)
        return self.log_dir

    def log_dir_exists(self):
        if os.path.exists(self.log_dir):
            return True
        return False


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
