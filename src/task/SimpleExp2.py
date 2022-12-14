import torch
import numpy as np

from task.EventMaker import EventMaker
from copy import deepcopy
from utils.utils import list_to_pth

t_query_test = [2,4,6]
t_query_feedback = [3,5,7]


class SimpleExp2():

    def __init__(self, B, T=5):
        self.T = T
        self.B = B
        self.em = EventMaker(B, T)
        self.x_dim = T+B
        self.y_dim = B
        self.stimuli_order = None
        # related parameters
        self.n_trios = B
        self.loc_sf_test = {'low d': 1, 'high d': 6}
        self.t_wtq = {'train': np.arange(0, self.T * 2, 2), 'test' : np.array([-1])}

    def sample_feature_values(self):
        return np.random.choice(range(0, self.B), size=(self.T-1,))

    def sample_shared_feature_loc(self):
        return np.random.choice(range(1, self.T))

    def sample_trial_type(self, p=.5):
        return 'low d' if np.random.uniform() < p else 'high d'

    def choose_within_trial_query(self, shared_feature_id):
        '''
        choose a time point from an event to query,
        but don't choose the shared feature or the event label at time 0
        return t
        '''
        assert 0 < shared_feature_id < self.T, 'shared feature cannot occur at time 0, and it cannot be T'
        valid_candidates = list(set(np.arange(1, self.T)).difference({shared_feature_id}))
        return np.random.choice(valid_candidates)

    def add_within_trial_query(self, event_matrix, shared_feature_id, test_mode=True):
        '''
        choose a time point from an event to query
        append a feature - value vector to the input event matrix
        '''
        if test_mode:
            # during test, don't choose the shared feature or the event label
            within_trial_query_id = self.choose_within_trial_query(shared_feature_id)
        else:
            # during training, anything can be queried
            within_trial_query_id = np.random.choice(np.arange(0, self.T))
        query_vector = deepcopy(event_matrix[within_trial_query_id,:])
        augmented_event_matrix = np.vstack([event_matrix, query_vector])
        return augmented_event_matrix, within_trial_query_id

    def interleave_within_trial_queries(self, event_matrix):
        perm = np.random.permutation(np.arange(0, self.T))
        all_queries = event_matrix[perm,:]
        return interleave_two_arrays(event_matrix, all_queries)

    def to_y(self, event_matrix):
        '''given an event matrix, such as 'targ_study'
        return the filler sub matrix (on the right hand side)
        '''
        return deepcopy(event_matrix[:, self.T:])

    def mask_out_query_fillers(self, test_event_matrix):
        test_event_matrix_cp = deepcopy(test_event_matrix)
        for i in t_query_test:
            test_event_matrix_cp[i,self.T:] = 0
        return test_event_matrix_cp

    def mask_out_within_trial_query_fillers(self, study_event_matrix, test_mode=True):
        study_event_matrix_cp = deepcopy(study_event_matrix)
        # remove the filler for the last row
        if test_mode:
            study_event_matrix_cp[-1, self.T:] = 0
        else:
            query_time_points_during_study = np.arange(1,self.T * 2, 2)
            study_event_matrix_cp[query_time_points_during_study, self.T:] = 0
        return study_event_matrix_cp

    def dup_query_timepoints(self, targ_test):
        '''
        0,1,2,3,4
        0,1,2,2,3,4 -> insert 3
        0,1,2,2,3,3,4 -> insert 5
        0,1,2,2,3,3,4,4 -> insert 7
        '''
        targ_test_ins = deepcopy(targ_test)
        for i in np.arange(2, self.T):
            insert_id = 2 * i - 1
            targ_test_ins = np.insert(targ_test_ins, insert_id, targ_test[i,:], axis=0)
        return targ_test_ins


    # def make_data(self, p=.5, n_tests=1, to_torch=False):
    #     """
    #     return a # events types (B) x |{targ, lure, targ_test}| x T x x_dim array
    #     plan:
    #         for each epoch, loop over all B events
    #             for each trial, the model sees targ, lure and then targ_test
    #                 then loop over time T
    #     """
    #     X, Y = [], []
    #     shared_feature_ids = [None] * self.B
    #     trial_types = [None] * self.B
    #     for event_label in range(self.B):
    #         feature_value_list = [event_label] + list(self.sample_feature_values())
    #         shared_feature_ids[event_label] = self.sample_shared_feature_loc()
    #         trial_types[event_label] = self.sample_trial_type(p)
    #         # make the target and lure for the study phase events  + the test target events
    #         targ_study, lure_study, targ_test = self.em.make_stimuli(
    #             feature_value_list, shared_feature_ids[event_label],
    #             trial_types[event_label], n_tests
    #         )
    #         # add within trial queries for the two study phase events
    #         targ_study, targ_query_id = self.add_within_trial_query(targ_study, shared_feature_ids[event_label])
    #         lure_study, lure_query_id = self.add_within_trial_query(lure_study, shared_feature_ids[event_label])
    #         # dup query time points (to provide feedback)
    #         targ_test = self.dup_query_timepoints(targ_test)
    #         # form Y here (before the fillers part of X are masked)
    #         Y.append([self.to_y(targ_study), self.to_y(lure_study), self.to_y(targ_test)])
    #         # remove filler values for the prediction features
    #         targ_study = self.mask_out_within_trial_query_fillers(targ_study)
    #         lure_study = self.mask_out_within_trial_query_fillers(lure_study)
    #         targ_test = self.mask_out_query_fillers(targ_test)
    #         # pack data
    #         X.append([targ_study, lure_study, targ_test])
    #     # see target -> lure -> test; since WM is flushed between events, order doesn't matter
    #     self.stimuli_order = ['targ', 'lure', 'test']
    #     # type conversion
    #     if to_torch:
    #         for j in range(self.B):
    #             X[j] = list_to_pth(X[j])
    #             Y[j] = list_to_pth(Y[j])
    #     return X, Y, shared_feature_ids, trial_types


    def make_data(self, p=.5, n_tests=1, test_mode=True, to_torch=False):
        """
        return a # events types (B) x |{targ, lure, targ_test}| x T x x_dim array
        plan:
            for each epoch, loop over all B events
                for each trial, the model sees targ, lure and then targ_test
                    then loop over time T
        """
        X, Y = [], []
        shared_feature_ids = [None] * self.B
        trial_types = [None] * self.B
        for event_label in range(self.B):
            feature_value_list = [event_label] + list(self.sample_feature_values())
            shared_feature_ids[event_label] = self.sample_shared_feature_loc()
            trial_types[event_label] = self.sample_trial_type(p)
            # make the target and lure for the study phase events  + the test target events
            targ_study, lure_study, targ_test = self.em.make_stimuli(
                feature_value_list, shared_feature_ids[event_label],
                trial_types[event_label], n_tests
            )
            if test_mode:
                # add within trial queries for the two study phase events
                targ_study, _ = self.add_within_trial_query(targ_study, shared_feature_ids[event_label])
                lure_study, _ = self.add_within_trial_query(lure_study, shared_feature_ids[event_label])
            else:
                targ_study = self.interleave_within_trial_queries(targ_study)
                lure_study = self.interleave_within_trial_queries(lure_study)

            # dup query time points (to provide feedback)
            targ_test = self.dup_query_timepoints(targ_test)
            # form Y here (before the fillers part of X are masked)
            Y.append([self.to_y(targ_study), self.to_y(lure_study), self.to_y(targ_test)])
            # remove filler values for the prediction features
            targ_study = self.mask_out_within_trial_query_fillers(targ_study, test_mode)
            lure_study = self.mask_out_within_trial_query_fillers(lure_study, test_mode)
            targ_test = self.mask_out_query_fillers(targ_test)
            # pack data
            X.append([targ_study, lure_study, targ_test])
        # see target -> lure -> test; since WM is flushed between events, order doesn't matter
        self.stimuli_order = ['targ', 'lure', 'test']
        # type conversion
        if to_torch:
            for j in range(self.B):
                X[j] = list_to_pth(X[j])
                Y[j] = list_to_pth(Y[j])
        return X, Y, shared_feature_ids, trial_types


def get_reward(a_t, x_t, y_t, penalty):
    """define the reward function at time t
    don't allow don't know response if it is an observation trial

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action
    penalty : int
        the penalty magnitude of making incorrect state prediction
    allow_dk : bool
        if True, then activating don't know makes r_t = 0, regardless of a_t

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    assert penalty >= 0
    dk_id = y_t.size()[0]
    a_t_targ = torch.argmax(y_t)
    # compare action vs. target action
    if is_query(x_t):
        if a_t == a_t_targ:
            r_t = 1
        elif a_t == dk_id:
            r_t = 0
        else:
            r_t = -penalty
    else:
        if a_t == a_t_targ:
            r_t = .1
        else:
            r_t = -penalty
    return torch.tensor(r_t, dtype=torch.float32)
    # return torch.from_numpy(np.array(r_t)).type(torch.FloatTensor).data


def is_query(x_t):
    if x_t.sum() == 1:
        # query trial
        return True
    elif x_t.sum() == 2:
        # copy trial
        return False
    else:
        raise ValueError('invalid x: must be 1 hot or 2 hot vector')


def add_query_indicator(x_t):
    assert torch.is_tensor(x_t)
    if is_query(x_t):
        return torch.cat([x_t, torch.ones(1)])
    else:
        return torch.cat([x_t, torch.zeros(1)])


def interleave_two_arrays(a, b, axis=0):
    '''demo
    a = np.zeros((3,3))
    b = np.ones((3,3))
    interleave_two_arrays(a, b)
    '''
    assert axis in [0, 1]
    n_rows_a, n_cols_a = np.shape(a)
    n_rows_b, n_cols_b = np.shape(b)
    if axis == 0:
        assert n_cols_a == n_cols_b
        c = np.empty((n_rows_a + n_rows_b, n_cols_a), dtype=a.dtype)
        c[0::2, :] = a
        c[1::2, :] = b
    else:
        assert n_rows_a == n_rows_b
        c = np.empty((n_rows_a, n_cols_a + n_cols_b), dtype=a.dtype)
        c[:, 0::2] = a
        c[:, 1::2] = b
    return c




if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as patches

    sns.set(style='white', palette='colorblind', context='poster')

    '''how to init'''
    B = 4
    test_mode = 0
    exp = SimpleExp2(B)

    '''core func'''

    targ_study, lure_study, targ_test = exp.em.make_stimuli([1,2,1,0,1], 3, 'low d')
    # plt.imshow(targ_test)
    # plt.imshow(targ_test_ins)

    '''the intended loop structure'''
    n_epochs = 1
    for i in range(n_epochs):
        X, Y, shared_feature_ids, trial_types = exp.make_data(test_mode=test_mode, to_torch=True)
        for j in range(exp.n_trios):
            print(f'Trio {j}')
            # np.random.permutation(X.shape[0])
            for k in range(len(exp.stimuli_order)): # loop over {targ, lure, targ_test}
                print(f'\tk = {k} - {exp.stimuli_order[k]}')
                for t in range(len(X[j][k])):
                    print(f'\t\tt = {t}, input shape = {np.shape(X[j][k][t])}, output shape = {np.shape(Y[j][k][t])}')

    '''visualize the data for 1 trial (a trio) '''
    for j in range(exp.n_trios):
        # j = 0
        shared_feature_ids_j = 2 * [shared_feature_ids[j]] + [exp.loc_sf_test[trial_types[j]]]
        f, axes = plt.subplots(3, 2, figsize=(10, 12), gridspec_kw={'height_ratios': [np.shape(X[j][k])[0] for k in range(len(X[j]))]})
        for k in range(3):

            axes[k, 0].imshow(X[j][k])
            axes[k, 0].set_title(f'X, {exp.stimuli_order[k]}')
            axes[k, 0].axvline(exp.T-.5, color='grey', linestyle='--')
            axes[k, 0].set_ylabel('time')
            axes[k, 1].imshow(Y[j][k])
            axes[k, 1].set_title(f'Y, {exp.stimuli_order[k]}')
            # mark the location of the shared feature
            if test_mode:
                rect = patches.Rectangle((0-.5, shared_feature_ids_j[k]-.5), exp.T+B, 1,
                    linewidth=3, edgecolor='red', facecolor='none')
                axes[k, 0].add_patch(rect)
                rect = patches.Rectangle((0-.5, shared_feature_ids_j[k]-.5), B, 1,
                    linewidth=3, edgecolor='red', facecolor='none')
                axes[k, 1].add_patch(rect)
        f.tight_layout()
