import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from task import SimpleExp2
from task import get_reward
from models import LCALSTM as Agent
from utils import Parameters as P
from utils.utils import to_sqnp, to_np, init_lll, save_ckpt
from utils.stats import compute_stats, entropy
# from models import LCALSTM_after as Agent
# torch.autograd.set_detect_anomaly(True)

sns.set(style='white', palette='colorblind', context='poster')

'''init params'''
# env param
B = 10
penalty = 4
# model param
add_query_indicator = True
gating_type = 'pre'
n_hidden = 128
lr = 1e-3
cmpt = .5
eta = 0.1
# training param
n_epochs = 4000
sup_epoch = 2000
test_mode = True
# save all params
p = P(
    B = B, penalty = penalty, n_hidden = n_hidden, lr = lr, cmpt = cmpt,
    eta = eta, test_mode = test_mode, add_query_indicator = add_query_indicator,
    gating_type = gating_type, n_epochs = n_epochs, sup_epoch = sup_epoch
)

'''init model and task'''
exp = SimpleExp2(B)

x_dim = exp.x_dim
y_dim = exp.y_dim + 1 # add 1 for the don't know unit
agent = Agent(
    input_dim=x_dim, output_dim=y_dim, rnn_hidden_dim=n_hidden,
    dec_hidden_dim=n_hidden//2, dict_len=2, cmpt=cmpt,
    add_query_indicator=add_query_indicator
)
# optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
optimizer_sup = torch.optim.Adam(agent.parameters(), lr=lr)
optimizer_rl = torch.optim.Adam(agent.parameters(), lr=lr)
# criterion = torch.nn.CrossEntropyLoss()

'''train the model'''
log_sf_ids = np.zeros((n_epochs, exp.n_trios))
log_trial_types = [None] * n_epochs
log_loss_s = torch.zeros((n_epochs, exp.n_trios, 3))
log_loss_a = torch.zeros((n_epochs, exp.n_trios, 3))
log_loss_c = torch.zeros((n_epochs, exp.n_trios, 3))
log_return = torch.zeros((n_epochs, exp.n_trios, 3))
log_acc = init_lll(n_epochs, exp.n_trios, 3)
log_a = init_lll(n_epochs, exp.n_trios, 3)
log_dk = init_lll(n_epochs, exp.n_trios, 3)
# i,j,k,t=0,0,0,0
for i in range(n_epochs):
    # make data
    X, Y, log_sf_ids[i], log_trial_types[i] = exp.make_data(test_mode=True, to_torch=True)
    permuted_tiro_ids = np.random.permutation(exp.n_trios)
    for j in permuted_tiro_ids:
        # go through the trio: event 1 (study) -> event 1' (study) -> event 1 (test)
        for k in range(len(exp.stimuli_order)):
            # at the beginning of each event, flush WM
            hc_t = agent.get_init_states()
            # turn off recall during the study phase, turn it on during test
            if k in [0, 1]:
                agent.retrieval_off()
            else:
                agent.retrieval_on()
            # init loss
            loss_sup, loss_rl, returns = 0, 0, 0
            agent.init_rvpe()
            for t in range(len(X[j][k])):
                # encode if and only if at event end
                if t == exp.T-1 and k in [0, 1]:
                    agent.encoding_on()
                else:
                    agent.encoding_off()
                # forward
                pi_a_t, v_t, hc_t, cache_t = agent(X[j][k][t], hc_t)
                a_t, p_a_t = agent.pick_action(pi_a_t)
                r_t = get_reward(a_t, X[j][k][t], Y[j][k][t], penalty)
                agent.append_rvpe(r_t, v_t, p_a_t, entropy(pi_a_t))
                # sup loss
                yhat_t = pi_a_t[:-1]
                loss_sup += F.mse_loss(yhat_t, Y[j][k][t])
                # loss_sup += criterion(yhat_t.view(1, -1), torch.argmax(Y[j][k][t]).view(-1,))
                # log info
                log_acc[i][j][k].append(int(torch.argmax(yhat_t) == torch.argmax(Y[j][k][t])))
                log_a[i][j][k].append(int(a_t))
                log_dk[i][j][k].append(int(a_t) == exp.B)

            # at the end of one event
            loss_actor, loss_critic, pi_ent = agent.compute_a2c_loss(gamma=0, normalize=True, use_V=True)

            # update weights
            if i < sup_epoch:
                optimizer_sup.zero_grad()
                loss_sup.backward()
                optimizer_sup.step()
            else:
                loss_rl = loss_actor + loss_critic - pi_ent * eta
                optimizer_rl.zero_grad()
                loss_rl.backward()
                optimizer_rl.step()

            # log info
            log_loss_s[i,j,k] = loss_sup
            log_loss_a[i,j,k], log_loss_c[i,j,k] = loss_actor, loss_critic
            log_return[i,j,k] = torch.stack(agent.rewards).sum()

        # at the end of 3 trio
        agent.flush_episodic_memory()

    # at the end of an epoch
    info_i = (i, log_loss_a[i].mean(), log_loss_c[i].mean(), log_return[i].mean())
    print(f'%3d | L: a: %.2f, c: %.2f | R : %.2f' % info_i)

# save weights
save_ckpt(n_epochs, p.gen_log_dirs(), agent, optimizer_rl)


'''preproc the results'''

def get_within_trial_query_mean(log_info):
    wtq_acc = np.zeros((n_epochs, exp.n_trios, 2))
    for i in range(n_epochs):
        for j in range(exp.n_trios):
            # get the last time point for the 1st two trios
            wtq_acc[i,j] = [log_info[i][j][k][-1] for k in range(2)]
    return np.mean(wtq_acc,axis=-1)

def get_test_query_mean(log_info):
    n_queries = 3
    tq_acc = np.zeros((n_epochs, exp.n_trios, n_queries))
    for i in range(n_epochs):
        for j in range(exp.n_trios):
            tq_acc[i,j] = [log_info[i][j][-1][ii] for ii in [2,4,6]]
    return tq_acc

def get_copy_mean(log_info):
    cp_acc = np.zeros((n_epochs, exp.n_trios, 2, exp.T))
    for i in range(n_epochs):
        for j in range(exp.n_trios):
            for k in range(2):
                cp_acc[i,j] = log_info[i][j][k][:exp.T]
    return np.mean(np.mean(cp_acc,axis=-1),axis=-1)


'''plot the results'''
def plot_learning_curve(title, data):
    f, ax = plt.subplots(1,1, figsize=(8,5))
    lc_data = [data[i].mean() for i in range(n_epochs)]
    ax.plot(lc_data)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(title)
    f.tight_layout()
    sns.despine()
    if sup_epoch < n_epochs:
        ax.axvline(sup_epoch, linestyle='--', color='red', label='start RL training')
    return f, ax

plot_learning_curve('Cumulative R', log_return)
plot_learning_curve('Loss - actor', log_loss_a)
plot_learning_curve('Loss - critic', log_loss_c)

chance = 1 / exp.B
wtq_acc = get_within_trial_query_mean(log_acc)
f, ax = plot_learning_curve('Within trial query acc', wtq_acc)
ax.axhline(chance, linestyle='--', color='grey', label='chance')
ax.legend()

tq_acc = get_test_query_mean(log_acc)
f, ax = plot_learning_curve('Test query acc', np.mean(tq_acc,axis=-1))
ax.axhline(chance, linestyle='--', color='grey', label='chance')
ax.legend()

cp_acc = get_copy_mean(log_acc)
f, ax = plot_learning_curve('Copy acc', np.mean(cp_acc,axis=-1))
ax.axhline(chance, linestyle='--', color='grey', label='chance')
ax.legend()


wtq_dk = get_within_trial_query_mean(log_dk)
f, ax = plot_learning_curve('Within trial query, p(dk)', wtq_dk)
ax.legend()

tq_dk = get_test_query_mean(log_dk)
f, ax = plot_learning_curve('Test query, p(dk)', np.mean(tq_dk,axis=-1))
ax.legend()

cp_dk = get_copy_mean(log_dk)
f, ax = plot_learning_curve('Copy, p(dk)', np.mean(cp_dk,axis=-1))
ax.legend()


# num of epochs to analyze
npa = 200
tq_dk_rs = np.reshape(tq_dk[-npa:], (-1, 3))
tq_acc_rs = np.reshape(tq_acc[-npa:], (-1, 3))
log_trial_types_rs = np.reshape(log_trial_types[-npa:], (-1))

hd = log_trial_types_rs == 'high d'
ld = log_trial_types_rs == 'low d'

tq_acc_rs_hd_mu, tq_acc_rs_hd_se = compute_stats(tq_acc_rs[hd],axis=0)
tq_acc_rs_ld_mu, tq_acc_rs_ld_se = compute_stats(tq_acc_rs[ld],axis=0)
tq_dk_rs_hd_mu, tq_dk_rs_hd_se = compute_stats(tq_dk_rs[hd],axis=0)
tq_dk_rs_ld_mu, tq_dk_rs_ld_se = compute_stats(tq_dk_rs[ld],axis=0)


c_pal = sns.color_palette('colorblind', n_colors=4)
alpha = .3
x_ = np.arange(3) # there are 3 query time points
ones = np.ones_like(x_)
f, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

axes[0].errorbar(x=x_, y=tq_acc_rs_hd_mu, yerr=tq_acc_rs_hd_se, color=c_pal[3])
axes[0].errorbar(x=x_, y=tq_acc_rs_ld_mu, yerr=tq_acc_rs_ld_se, color=c_pal[0])
axes[0].set_ylabel('correct')
axes[1].errorbar(x=x_, y=tq_dk_rs_hd_mu, yerr=tq_dk_rs_hd_se, color=c_pal[3])
axes[1].errorbar(x=x_, y=tq_dk_rs_ld_mu, yerr=tq_dk_rs_ld_se, color=c_pal[0])
axes[1].set_ylabel('dont know')
axes[2].errorbar(x=x_, y=1 - tq_acc_rs_hd_mu - tq_dk_rs_hd_mu, yerr=tq_acc_rs_hd_se, color=c_pal[3])
axes[2].errorbar(x=x_, y=1 - tq_acc_rs_ld_mu - tq_dk_rs_ld_mu, yerr=tq_acc_rs_ld_se, color=c_pal[0])
axes[2].set_ylabel('incorrect')

for ax in axes:
    ax.set_xlabel('Test query position')
    ax.set_xticks(x_)
    ax.set_xticklabels(x_+2)
f.legend(['high d', 'low d'], loc=(.2,.25))
sns.despine()
f.tight_layout()
