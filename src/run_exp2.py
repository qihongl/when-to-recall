import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
# from utils import to_pth
from task import SimpleExp2
from models import LCALSTM as Agent
from models import get_reward, compute_returns, compute_a2c_loss

torch.autograd.set_detect_anomaly(True)

sns.set(style='white', palette='colorblind', context='poster')


'''init task'''
T = 5
B = 4
penalty = 1
exp = SimpleExp2(T,B)

'''init model'''
n_hidden = 64
lr = 1e-3
cmpt = .5
x_dim = exp.x_dim
# x_dim = exp.x_dim + 1 # add 1 for the penalty indicator
y_dim = exp.y_dim + 1 # add 1 for the penalty indicator
agent = Agent(
    input_dim=x_dim, output_dim=y_dim, rnn_hidden_dim=n_hidden,
    dec_hidden_dim=n_hidden, dict_len=2, cmpt=cmpt
)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
# optimizer_sup = torch.optim.Adam(agent.parameters(), lr=lr)
# scheduler_sup = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_sup, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True)
# optimizer_rl = torch.optim.Adam(agent.parameters(), lr=lr)
# scheduler_rl = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_rl, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True)


'''train the model'''
n_epochs = 1000
log_sf_ids = np.zeros((n_epochs, exp.n_trios))
log_trial_types = [None] * n_epochs
log_wtq_ids = np.zeros((n_epochs, exp.n_trios, 2))
log_loss_a = torch.zeros((n_epochs, exp.n_trios, 3))
log_loss_c = torch.zeros((n_epochs, exp.n_trios, 3))
log_return = torch.zeros((n_epochs, exp.n_trios, 3))
for i in range(n_epochs):
    # print(f'Epoch {i}')
    # make data
    X, Y, log_sf_ids[i], log_trial_types[i], log_wtq_ids[i] = exp.make_data(to_torch=True)
    permuted_tiro_ids = np.random.permutation(exp.n_trios)
    for j in permuted_tiro_ids:
        # print(f'\tTrio {j}')
        # go through the trio: event 1 (study) -> event 1' (study) -> event 1 (test)
        for k in range(len(exp.stimuli_order)):
            # print(f'\t\tk = {k} - {exp.stimuli_order[k]}')
            # at the beginning of each event, flush WM
            hc_t = agent.get_init_states()
            # turn off recall during the study phase, turn it on during test
            if k in [0, 1]:
                agent.retrieval_off()
            else:
                agent.retrieval_on()
            # init loss
            loss_sup, loss_rl, returns = 0, 0, 0
            rewards, values, probs = [], [], []
            for t in range(T):
                # print(f'\t\t\tt = {t}, input shape = {np.shape(X[j][k][t])}, output shape = {np.shape(Y[j][k][t])}')
                # encode if and only if at event end
                if t == T-1:
                    agent.encoding_on()
                else:
                    agent.encoding_off()
                # forward
                pi_a_t, v_t, hc_t, cache_t = agent(X[j][k][t].view(1, 1, -1), hc_t)
                a_t, p_a_t = agent.pick_action(pi_a_t)
                r_t = get_reward(a_t, Y[j][k][t], penalty)

                # collect results
                rewards.append(r_t)
                values.append(v_t)
                probs.append(pi_a_t)
                # # sup loss
                # yhat_t = torch.squeeze(pi_a_t)[:-1]
                # loss_sup += F.mse_loss(yhat_t, Y[j][k][t])

            returns = compute_returns(rewards)
            loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
            loss_rl = loss_actor + loss_critic
            # at the end of one event
            optimizer.zero_grad()
            # loss_sup.backward()
            loss_rl.backward()
            optimizer.step()

            # log info
            log_loss_a[i,j,k], log_loss_c[i,j,k] = loss_actor, loss_critic
            log_return[i,j,k] = torch.stack(rewards).sum()

        # at the end of 3 trio
        agent.flush_episodic_memory()

    # at the end of an epoch
    print(f'Epoch {i} | L: c: %.2f, a: %.2f | R : %.2f' % (log_loss_a[i].mean(), log_loss_c[i].mean(), log_return[i].mean()))


'''preproc the results'''

'''plot the results'''
f, ax = plt.subplots(1,1, figsize=(8,5))
lc_return = [log_return[i].mean() for i in range(n_epochs)]
ax.plot(lc_return)
ax.set_xlabel('Epochs')
ax.set_ylabel('Cumulative R')
f.tight_layout()
sns.despine()
