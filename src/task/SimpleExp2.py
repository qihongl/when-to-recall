import torch
import numpy as np
from task.EventMaker import EventMaker
from utils import to_pth

class SimpleExp2():

    def __init__(self, T, B):
        self.T = T
        self.B = B
        self.em = EventMaker(T, B)
        self.x_dim = T+B
        self.y_dim = B

    def sample_feature_values(self):
        return np.random.choice(range(0, self.B), size=(self.T-1,))

    def sample_shared_feature_loc(self):
        return np.random.choice(range(1, self.T))

    def sample_trial_type(self, p=.5):
        return 'low d' if np.random.uniform() < p else 'high d'

    def make_data(self, to_torch=False):
        """
        return a # events types (B) x |{targ, lure, targ_test}| x T x x_dim array
        plan:
            for each epoch, loop over all B events
                for each trial, the model sees targ, lure and then targ_test
                    then loop over time T
        """
        p = .5 # TODO
        X = []
        for event_label in range(self.B):
            feature_value_list = [event_label] + list(self.sample_feature_values())
            shared_feature_id = self.sample_shared_feature_loc()
            trial_type = self.sample_trial_type(p)
            # make the target and lure for the study phase events  + the test target events
            targ_study, lure_study, targ_test = self.em.make_stimuli(feature_value_list, shared_feature_id, trial_type)
            # see target -> lure -> test; since WM is flushed between events, order doesn't matter
            X_i = np.stack([targ_study, lure_study, targ_test])
            X.append(X_i)
        X = np.array(X)
        # type conversion
        if to_torch:
            X = to_pth(X)
        return X



if __name__ == "__main__":
    '''how to use'''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as patches

    sns.set(style='white', palette='colorblind', context='poster')

    # init
    T = 5
    B = 4
    exp = SimpleExp2(T,B)

    n_epochs = 1
    for i in range(n_epochs):
        X = exp.make_data()
        print(X.shape)
        for j in range(3): # loop over {targ, lure, targ_test}
            for t in range(T):
                print(np.shape(X[i,j,t]))
