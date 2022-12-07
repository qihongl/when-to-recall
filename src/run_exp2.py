import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from utils import to_sqnp, to_np, init_lll
from stats import compute_stats, entropy
from task import SimpleExp2
from models import LCALSTM as Agent
from models import get_reward, compute_returns, compute_a2c_loss
# torch.autograd.set_detect_anomaly(True)

sns.set(style='white', palette='colorblind', context='poster')


'''init task'''
T = 5
B = 10
penalty = 0
exp = SimpleExp2(T, B)

'''init model'''
n_hidden = 64
lr = 1e-3
cmpt = .5
eta = 0
x_dim = exp.x_dim
# x_dim = exp.x_dim + 1 # add 1 for the penalty indicator
y_dim = exp.y_dim + 1 # add 1 for the penalty indicator
agent = Agent(
    input_dim=x_dim, output_dim=y_dim, rnn_hidden_dim=n_hidden,
    dec_hidden_dim=n_hidden, dict_len=2, cmpt=cmpt
)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
optimizer_sup = torch.optim.Adam(agent.parameters(), lr=lr)
optimizer_rl = torch.optim.Adam(agent.parameters(), lr=lr)
# scheduler_sup = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_sup, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True)
# scheduler_rl = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer_rl, factor=1/2, patience=30, threshold=1e-3, min_lr=1e-8,
#     verbose=True)


'''train the model'''
n_epochs = 3000
sup_epoch = 2000
log_sf_ids = np.zeros((n_epochs, exp.n_trios))
log_trial_types = [None] * n_epochs
log_wtq_ids = np.zeros((n_epochs, exp.n_trios, 2))
log_loss_s = torch.zeros((n_epochs, exp.n_trios, 3))
log_loss_a = torch.zeros((n_epochs, exp.n_trios, 3))
log_loss_c = torch.zeros((n_epochs, exp.n_trios, 3))
log_return = torch.zeros((n_epochs, exp.n_trios, 3))
log_acc = init_lll(n_epochs, exp.n_trios, 3)
log_a = init_lll(n_epochs, exp.n_trios, 3)
log_dk = init_lll(n_epochs, exp.n_trios, 3)


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
            rewards, values, probs, ents, accs = [], [], [], [], []
            for t in range(len(X[j][k])):
                # print(f'\t\t\tt = {t}, input shape = {np.shape(X[j][k][t])}, output shape = {np.shape(Y[j][k][t])}')
                # encode if and only if at event end
                if t == T-1:
                    agent.encoding_on()
                else:
                    agent.encoding_off()
                # forward
                pi_a_t, v_t, hc_t, cache_t = agent(X[j][k][t], hc_t)
                a_t, p_a_t = agent.pick_action(pi_a_t)
                r_t = get_reward(a_t, X[j][k][t], Y[j][k][t], penalty)

                # collect results
                rewards.append(r_t)
                values.append(v_t)
                probs.append(pi_a_t)
                ents.append(entropy(pi_a_t))
                # sup loss
                yhat_t = pi_a_t[:-1]
                loss_sup += F.mse_loss(yhat_t, Y[j][k][t])
                # log info
                log_acc[i][j][k].append(int(torch.argmax(yhat_t) == torch.argmax(Y[j][k][t])))
                log_a[i][j][k].append(int(a_t))
                log_dk[i][j][k].append(int(a_t) == exp.B)

            returns = compute_returns(rewards, normalize=True)
            loss_actor, loss_critic = compute_a2c_loss(probs, values, returns)
            pi_ent = torch.stack(ents).sum()

            loss_rl = loss_actor + loss_critic - pi_ent * eta
            # at the end of one event

            if i < sup_epoch:
                optimizer_sup.zero_grad()
                loss_sup.backward()
                optimizer_sup.step()
            else:
                optimizer_rl.zero_grad()
                loss_rl.backward()
                optimizer_rl.step()

            # log info
            log_loss_s[i,j,k] = loss_sup
            log_loss_a[i,j,k], log_loss_c[i,j,k] = loss_actor, loss_critic
            log_return[i,j,k] = torch.stack(rewards).sum()

        # at the end of 3 trio
        agent.flush_episodic_memory()

    # at the end of an epoch
    info_i = (i, log_loss_a[i].mean(), log_loss_c[i].mean(), log_return[i].mean())
    print(f'%3d | L: a: %.2f, c: %.2f | R : %.2f' % info_i)


'''preproc the results'''

def get_within_trial_query_accuracy(log_acc):
    wtq_acc = np.zeros((n_epochs, exp.n_trios, 2))
    for i in range(n_epochs):
        for j in range(exp.n_trios):
            # get the last time point for the 1st two trios
            wtq_acc[i,j] = [log_acc[i][j][k][-1] for k in range(2)]
    return np.mean(wtq_acc,axis=-1)

def get_test_query_acc(log_acc):
    n_queries = 3
    tq_acc = np.zeros((n_epochs, exp.n_trios, n_queries))
    for i in range(n_epochs):
        for j in range(exp.n_trios):
            tq_acc[i,j] = log_acc[i][j][-1][exp.T - n_queries:]
    return tq_acc

def get_copy_acc(log_acc):
    cp_acc = np.zeros((n_epochs, exp.n_trios, 2, exp.T))
    for i in range(n_epochs):
        for j in range(exp.n_trios):
            for k in range(2):
                cp_acc[i,j] = log_acc[i][j][k][:exp.T]
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
wtq_acc = get_within_trial_query_accuracy(log_acc)
f, ax = plot_learning_curve('Within trial query acc', wtq_acc)
ax.axhline(chance, linestyle='--', color='grey', label='chance')
ax.legend()

tq_acc = get_test_query_acc(log_acc)
f, ax = plot_learning_curve('Test query acc', np.mean(tq_acc,axis=-1))
ax.axhline(chance, linestyle='--', color='grey', label='chance')
ax.legend()

cp_acc = get_copy_acc(log_acc)
f, ax = plot_learning_curve('Copy acc', np.mean(cp_acc,axis=-1))
ax.axhline(chance, linestyle='--', color='grey', label='chance')
ax.legend()


wtq_dk = get_within_trial_query_accuracy(log_dk)
f, ax = plot_learning_curve('Within trial query, p(dk)', wtq_dk)
ax.legend()

tq_dk = get_test_query_acc(log_dk)
f, ax = plot_learning_curve('Test query, p(dk)', np.mean(tq_dk,axis=-1))
ax.legend()

cp_dk = get_copy_acc(log_dk)
f, ax = plot_learning_curve('Copy, p(dk)', np.mean(cp_dk,axis=-1))
ax.legend()
