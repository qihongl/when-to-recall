import torch
import numpy as np

from task.EventMaker import EventMaker
from copy import deepcopy
from utils import list_to_pth

class SimpleExp2():

    def __init__(self, T, B):
        self.T = T
        self.B = B
        self.em = EventMaker(T, B)
        self.x_dim = T+B
        self.y_dim = B
        self.stimuli_order = None
        # related parameters
        self.n_trios = B

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
        test_event_matrix_cp[2:,self.T:] = 0
        return test_event_matrix_cp

    def mask_out_within_trial_query_fillers(self, study_event_matrix):
        study_event_matrix_cp = deepcopy(study_event_matrix)
        study_event_matrix_cp[-1,self.T:] = 0
        return study_event_matrix_cp



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
            # form Y
            Y.append([self.to_y(targ_study), self.to_y(lure_study), self.to_y(targ_test)])
            # remove filler values for the prediction features
            targ_study = self.mask_out_within_trial_query_fillers(targ_study)
            lure_study = self.mask_out_within_trial_query_fillers(lure_study)
            targ_test = self.mask_out_query_fillers(targ_test)
            # pack data
            X.append([targ_study, lure_study, targ_test])

            # print(np.shape(targ_study))
            # print(np.shape(lure_study))
            # print(np.shape(targ_test))

        # see target -> lure -> test; since WM is flushed between events, order doesn't matter
        self.stimuli_order = ['targ', 'lure', 'test']
        within_trial_query_ids = np.array(within_trial_query_ids)
        # # only need to output the feature value
        # Y = deepcopy(X[:,:,:,self.T:])
        # # mask out the feature values for the queries during test
        # X[:,-1, 2:,self.T:] = 0

        # type conversion
        if to_torch:
            for j in range(self.B):
                X[j] = list_to_pth(X[j])
                Y[j] = list_to_pth(Y[j])
        return X, Y, shared_feature_ids, trial_types, within_trial_query_ids



if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as patches

    sns.set(style='white', palette='colorblind', context='poster')

    '''how to init'''
    T = 5
    B = 4
    exp = SimpleExp2(T,B)
    # X, Y, shared_feature_ids, trial_types, within_trial_query_ids = exp.make_data()
    # np.shape(X)
    # X[0][0]

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
    shared_feature_ids_j = 2 * [shared_feature_ids[j]] + [exp.em.loc_sf_test[trial_types[j]]]
    f, axes = plt.subplots(3, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [np.shape(X[j][k])[0] for k in range(len(X[j]))]})
    for k in range(3):

        axes[k, 0].imshow(X[j][k])
        axes[k, 0].set_title(f'X, {exp.stimuli_order[k]}')
        axes[k, 0].axvline(T-.5, color='grey', linestyle='--')
        axes[k, 0].set_ylabel('time')
        # mark the location of the shared feature
        rect = patches.Rectangle((0-.5, shared_feature_ids_j[k]-.5), T+B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[k, 0].add_patch(rect)

        axes[k, 1].imshow(Y[j][k])
        axes[k, 1].set_title(f'Y, {exp.stimuli_order[k]}')
        rect = patches.Rectangle((0-.5, shared_feature_ids_j[k]-.5), B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[k, 1].add_patch(rect)
    f.tight_layout()
