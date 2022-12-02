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
    print(X.shape)
    for j in range(3): # loop over {targ, lure, targ_test}
        for t in range(T):
            print(np.shape(X[i,j,t]))
