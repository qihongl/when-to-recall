import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from utils import to_pth
from task import SimpleExp2
from models import LCALSTM as Agent


sns.set(style='white', palette='colorblind', context='poster')


'''init task'''
T = 5
B = 4
exp = SimpleExp2(T,B)

'''init model'''
n_hidden = 64
lr = 1e-3
cmpt = .5

agent = Agent(
    input_dim=exp.x_dim, output_dim=exp.y_dim, rnn_hidden_dim=n_hidden,
    dec_hidden_dim=n_hidden, dict_len=3, cmpt=cmpt
)

optimizer_sup = torch.optim.Adam(agent.parameters(), lr=lr)
# scheduler_sup = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_sup, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True)

optimizer_rl = torch.optim.Adam(agent.parameters(), lr=lr)
# scheduler_rl = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_rl, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True)


'''train the model'''
n_epochs = 1
for i in range(n_epochs):
    X, Y = exp.make_data(to_torch=True)
    n_trials = X.shape[0]
    for j in range(n_trials):
        print(f'Trial {j}')
        np.random.permutation(X.shape[0])
        for k in range(len(exp.stimuli_order)): # loop over {targ, lure, targ_test}
            print(f'\tk = {k} - {exp.stimuli_order[k]}')
            # at the beginning of each sub-trial, flush WM
            hc_t = agent.get_init_states()
            for t in range(T):
                print(f'\t\tt = {t}, input shape = {np.shape(X[j,k,t])}')
                # forward
                pi_a_t, v_t, hc_t, cache_t = agent(X[j,k,t].view(1, 1, -1), hc_t)

            # optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.parameters(), 1)
            # optimizer.step()

'''preproc the results'''

'''plot the results'''
