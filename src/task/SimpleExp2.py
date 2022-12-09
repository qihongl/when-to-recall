import torch
import numpy as np

from task.EventMaker import EventMaker
from copy import deepcopy
from utils import list_to_pth

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

    def add_within_trial_query(self, event_matrix, shared_feature_id):
        '''
        choose a time point from an event to query
        append a feature - value vector to the input event matrix
        '''
        within_trial_query_id = self.choose_within_trial_query(shared_feature_id)
        query_vector = deepcopy(event_matrix[within_trial_query_id,:])
        # query_vector[self.T:] = 0
        augmented_event_matrix = np.vstack([event_matrix, query_vector])
        return augmented_event_matrix, within_trial_query_id

    def to_y(self, event_matrix):
        '''given an event matrix, such as 'targ_study'
        return the filler sub matrix (on the right hand side)
        '''
        return deepcopy(event_matrix[:, self.T:])

    def mask_out_query_fillers(self, test_event_matrix):
        test_event_matrix_cp = deepcopy(test_event_matrix)
        for i in [2,4,6]:
            test_event_matrix_cp[i,self.T:] = 0
        return test_event_matrix_cp

    def mask_out_within_trial_query_fillers(self, study_event_matrix):
        study_event_matrix_cp = deepcopy(study_event_matrix)
        study_event_matrix_cp[-1,self.T:] = 0
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


    def make_data(self, to_torch=False):
        """
        return a # events types (B) x |{targ, lure, targ_test}| x T x x_dim array
        plan:
            for each epoch, loop over all B events
                for each trial, the model sees targ, lure and then targ_test
                    then loop over time T
        """
        p = .5 # TODO
        X, Y = [], []
        shared_feature_ids = [None] * self.B
        trial_types = [None] * self.B
        within_trial_query_ids = [None] * self.B
        for event_label in range(self.B):
            feature_value_list = [event_label] + list(self.sample_feature_values())
            shared_feature_ids[event_label] = self.sample_shared_feature_loc()
            trial_types[event_label] = self.sample_trial_type(p)
            # make the target and lure for the study phase events  + the test target events
            targ_study, lure_study, targ_test = self.em.make_stimuli(
                feature_value_list, shared_feature_ids[event_label], trial_types[event_label]
            )
            # add within trial queries for the two study phase events
            targ_study, targ_query_id = self.add_within_trial_query(targ_study, shared_feature_ids[event_label])
            lure_study, lure_query_id = self.add_within_trial_query(lure_study, shared_feature_ids[event_label])
            within_trial_query_ids[event_label] = [targ_query_id, lure_query_id]
            #
            targ_test = self.dup_query_timepoints(targ_test)
            # form Y
            Y.append([self.to_y(targ_study), self.to_y(lure_study), self.to_y(targ_test)])
            # remove filler values for the prediction features
            targ_study = self.mask_out_within_trial_query_fillers(targ_study)
            lure_study = self.mask_out_within_trial_query_fillers(lure_study)
            targ_test = self.mask_out_query_fillers(targ_test)
            # pack data
            X.append([targ_study, lure_study, targ_test])
        # see target -> lure -> test; since WM is flushed between events, order doesn't matter
        self.stimuli_order = ['targ', 'lure', 'test']
        # type conversion
        within_trial_query_ids = np.array(within_trial_query_ids)
        if to_torch:
            for j in range(self.B):
                X[j] = list_to_pth(X[j])
                Y[j] = list_to_pth(Y[j])
        return X, Y, shared_feature_ids, trial_types, within_trial_query_ids


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
            r_t = -1
    else:
        if a_t == a_t_targ:
            r_t = .1
        else:
            r_t = -1
    return torch.from_numpy(np.array(r_t)).type(torch.FloatTensor).data


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


if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as patches

    sns.set(style='white', palette='colorblind', context='poster')

    '''how to init'''
    B = 4
    exp = SimpleExp2(B)

    '''core func'''

    targ_study, lure_study, targ_test = exp.em.make_stimuli([1,2,1,0,1], 3, 'low d')
    # plt.imshow(targ_test)
    # plt.imshow(targ_test_ins)


    '''the intended loop structure'''
    n_epochs = 1
    for i in range(n_epochs):
        X, Y, shared_feature_ids, trial_types, within_trial_query_ids = exp.make_data(to_torch=True)
        for j in range(exp.n_trios):
            print(f'Trio {j}')
            # np.random.permutation(X.shape[0])
            for k in range(len(exp.stimuli_order)): # loop over {targ, lure, targ_test}
                print(f'\tk = {k} - {exp.stimuli_order[k]}')
                for t in range(len(X[j][k])):
                    print(f'\t\tt = {t}, input shape = {np.shape(X[j][k][t])}, output shape = {np.shape(Y[j][k][t])}')

    '''visualize the data for 1 trial (a trio) '''
    j = 0
    shared_feature_ids_j = 2 * [shared_feature_ids[j]] + [exp.loc_sf_test[trial_types[j]]]
    f, axes = plt.subplots(3, 2, figsize=(10, 12), gridspec_kw={'height_ratios': [np.shape(X[j][k])[0] for k in range(len(X[j]))]})
    for k in range(3):

        axes[k, 0].imshow(X[j][k])
        axes[k, 0].set_title(f'X, {exp.stimuli_order[k]}')
        axes[k, 0].axvline(exp.T-.5, color='grey', linestyle='--')
        axes[k, 0].set_ylabel('time')
        # mark the location of the shared feature
        rect = patches.Rectangle((0-.5, shared_feature_ids_j[k]-.5), exp.T+B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[k, 0].add_patch(rect)

        axes[k, 1].imshow(Y[j][k])
        axes[k, 1].set_title(f'Y, {exp.stimuli_order[k]}')
        rect = patches.Rectangle((0-.5, shared_feature_ids_j[k]-.5), B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[k, 1].add_patch(rect)
    f.tight_layout()
