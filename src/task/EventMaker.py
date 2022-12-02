import numpy as np
from copy import deepcopy
from utils import onehot

''' doc
Make an event, which is a sequence of feature vectors

Example:
    Study trials:
        E : 1 A B C D
        E': 1 A E F G
    Test trials:
        1 A C D B (low diagnosticity, know it is E @ t = 2)
        1 C D B A (high diagnosticity, know it is E @ t = 1)

Properties of the task:
1. the 1st feature vector indicates the event type,
    which is always shared between A and A', where A and A' are two events
2. In Cody's design, event length = 5, and other than the 1st feature,
    one of the feature is shared across A and A'
3. Diagnosticity is determined by the location
'''

class EventMaker():
    """
    Parameters
    ----------
    T : int
        event length
    B : int
        the branching factor - the # of feature values

    Properties:
    1. dim(feature vector) = T + B
    2. # possible events = B^T.
        For example, if T = 5, B = 3, then # possible events = 3^5 = 243
    3. If each event label must correspond to a unique set of feature values, then we can have B trials in total.
        for each epoch, we can reconstruct each trial, within each epoch, # possible events = B^(T-1)
    """

    def __init__(self, T=5, B=3):
        self.T = T
        self.B = B
        self.x_dim = T + B
        self.loc_sf_test = {'low d': 1, 'high d':T-1}

    def _feature_value_vectors(self, feature_value_list):
        """make a trial

        Parameters
        ----------
        feature_value_list : 1d np array
            specifies the feature value at time t, for all t

        Returns
        -------
        type
            a T x B matrix

        """
        assert len(feature_value_list) == self.T
        assert np.all(feature_value_list) < self.B
        return np.array([onehot(self.B, fv) for t, fv in enumerate(feature_value_list)])

    def make_event(self, feature_value_list):
        fvm = self._feature_value_vectors(feature_value_list)
        return np.hstack([np.eye(self.T), fvm])

    def make_lure_feature_value_list(self, feature_value_list, shared_feature_id):
        """
        *the 0th entry is the event label, so it doesn't get altered
        """
        assert 0 < shared_feature_id < self.T, 'shared feature cannot occur at time 0, and it cannot be T'
        lure_feature_value_list = deepcopy(feature_value_list)
        for t in np.arange(1, self.T):
            if t == shared_feature_id:
                continue
            v_to_exclude = {feature_value_list[t]}
            alternative_vs = {i for i in range(self.B)}.difference(v_to_exclude)
            lure_feature_value_list[t] = np.random.choice(list(alternative_vs))
        return lure_feature_value_list


    def make_targ_lure(self, feature_value_list, shared_feature_id):
        lure_feature_value_list = self.make_lure_feature_value_list(feature_value_list, shared_feature_id)
        targ = self.make_event(feature_value_list)
        lure = self.make_event(lure_feature_value_list)
        return targ, lure

    def make_low_d_stimuli(self, feature_value_list, shared_feature_id):
        '''
        For a low diagnosticity trial (where the diagnostic feature comes at position 3)
        The second observation is always a shared feature.
        '''
        targ_study, lure_study = self.make_targ_lure(feature_value_list, shared_feature_id)
        # make the test trial
        targ_test = np.zeros_like(targ_study)
        # the 1st feature is always the event label
        targ_test[0,:] = targ_study[0,:]
        # put the shared feature right after the event label, permute the rest of the features
        targ_test[self.loc_sf_test['low d'],:] = targ_study[shared_feature_id,:]
        # permute the rest of the features
        rest_feature_ids = {i for i in range(1, self.T)}.difference({shared_feature_id})
        rest_feature_ids_perm = np.random.permutation(list(rest_feature_ids))
        for i in range(2, self.T):
            targ_test[i,:] = targ_study[rest_feature_ids_perm[i-2],:]
        return targ_study, lure_study, targ_test

    def make_high_d_stimuli(self, feature_value_list, shared_feature_id):
        '''
        For a low diagnosticity trial (where the diagnostic feature comes at position 3)
        The second observation is always a shared feature.
        '''
        # shared_feature_id_test = -1
        targ_study, lure_study = self.make_targ_lure(feature_value_list, shared_feature_id)
        # make the test trial
        targ_test = np.zeros_like(targ_study)
        # the 1st feature is always the event label
        targ_test[0,:] = targ_study[0,:]
        # put the shared feature right after the event label, permute the rest of the features
        targ_test[self.loc_sf_test['high d'],:] = targ_study[shared_feature_id,:]
        # permute the rest of the features
        rest_feature_ids = {i for i in range(1, self.T)}.difference({shared_feature_id})
        rest_feature_ids_perm = np.random.permutation(list(rest_feature_ids))
        for i in range(1, self.T-1):
            targ_test[i,:] = targ_study[rest_feature_ids_perm[i-1],:]
        return targ_study, lure_study, targ_test


    def make_stimuli(self, feature_value_list, shared_feature_id, trial_type):
        assert trial_type in ['low d', 'high d']
        if trial_type == 'low d':
            return self.make_low_d_stimuli(feature_value_list, shared_feature_id)
        return self.make_high_d_stimuli(feature_value_list, shared_feature_id)


    def feature_vec_to_ints(self, feature_vector):
        '''
        given a T+B dim feature vector
        return t and b
        '''
        assert len(feature_vector) == self.T + self.B
        feature_vector_T = feature_vector[:self.T]
        feature_vector_B = feature_vector[self.T:]
        assert sum(feature_vector_T) == sum(feature_vector_B) == 1
        t = np.where(np.array(feature_vector_T))[0][0]
        b = np.where(np.array(feature_vector_B))[0][0]
        return np.array(t, b)

    def feature_vecs_to_ints(self, feature_vectors):
        assert len(feature_vectors.shape) == 2
        return np.array([feature_vec_to_ints(fv) for fv in feature_vectors])

# def imshow_stimuli(targ_study, lure_study, targ_test):


if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as patches

    sns.set(style='white', palette='colorblind', context='poster')

    # init
    T = 5
    B = 3
    em = EventMaker(T,B)

    # provide the list of feature values, and the location of the shared feature
    feature_value_list = np.array([1,2,1,0,1])
    shared_feature_id = 4
    lure_feature_value_list = em.make_lure_feature_value_list(feature_value_list, shared_feature_id)
    print(f'targ  = {feature_value_list}')
    print(f'lure = {lure_feature_value_list}')
    print(f'match = {feature_value_list == lure_feature_value_list}')

    '''make a low d trial'''
    trial_types = ['low d', 'high d']
    for trial_type in trial_types:

        # targ, lure = em.make_targ_lure(feature_value_list, shared_feature_id)
        targ_study, lure_study, targ_test = em.make_stimuli(feature_value_list, shared_feature_id, trial_type)
        f, axes = plt.subplots(3,1, figsize=(10, 10))
        axes[0].imshow(targ_study)
        axes[0].set_title('targ, study')
        axes[0].set_ylabel('time')
        axes[1].imshow(lure_study)
        axes[1].set_title('lure, study')
        axes[1].set_ylabel('time')
        axes[2].imshow(targ_test)
        axes[2].set_title('targ, test')
        axes[2].set_ylabel('time')
        for ax in axes:
            ax.axvline(T-.5, color='grey', linestyle='--')
        # mark the location of the shared feature
        rect = patches.Rectangle((0-.5, shared_feature_id-.5), T+B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        rect = patches.Rectangle((0-.5, shared_feature_id-.5), T+B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        rect = patches.Rectangle((0-.5, em.loc_sf_test[trial_type]-.5), T+B, 1, linewidth=3, edgecolor='red', facecolor='none')
        axes[2].add_patch(rect)
        f.tight_layout()

    '''question: does ths subject see the answer once they respond to the query regardless of whether they got it right? '''
