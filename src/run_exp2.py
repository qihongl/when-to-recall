import torch
import numpy as np
# from utils import to_pth
from task import SimpleExp2
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', palette='colorblind', context='poster')

# init
T = 5
B = 4
exp = SimpleExp2(T,B)

n_epochs = 1
for i in range(n_epochs):
    X = exp.make_data()
    n_trials = X.shape[0]
    for j in range(n_trials):
        print(f'Trial {j}')
        np.random.permutation(X.shape[0])
        for k in range(3): # loop over {targ, lure, targ_test}
            print(f'\tk = {k}')
            for t in range(T):
                print(f'\t\tt = {t}, input shape = {np.shape(X[j,k,t])}')
