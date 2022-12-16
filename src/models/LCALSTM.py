import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from models.EM import EM
from task import add_query_indicator
from torch.distributions import Categorical
from models.initializer import initialize_weights
from torch.nn.functional import smooth_l1_loss

# constants
# number of vector signal (lstm gates)
N_VSIG = 3
# number of scalar signal (sigma)
N_SSIG = 1
# the ordering in the cache
scalar_signal_names = ['input strength']
vector_signal_names = ['f', 'i', 'o']
eps = np.finfo(np.float32).eps.item()
sigmoid = nn.Sigmoid()


class LCALSTM(nn.Module):

    def __init__(
            self, input_dim, output_dim, rnn_hidden_dim, dec_hidden_dim,
            kernel='cosine', dict_len=2, weight_init_scheme='ortho', cmpt=.8,
            add_query_indicator=False,
    ):
        super(LCALSTM, self).__init__()
        self.cmpt = cmpt
        self.add_query_indicator = add_query_indicator
        if add_query_indicator:
            self.input_dim = input_dim+1
        else:
            self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_hidden_total = (N_VSIG + 1) * rnn_hidden_dim + N_SSIG
        # rnn module
        self.i2h = nn.Linear(self.input_dim, self.n_hidden_total)
        self.h2h = nn.Linear(rnn_hidden_dim, self.n_hidden_total)
        # deicion module
        self.ih = nn.Linear(rnn_hidden_dim, dec_hidden_dim)
        self.actor = nn.Linear(dec_hidden_dim, output_dim)
        self.critic = nn.Linear(dec_hidden_dim, 1)
        # memory
        self.hpc = nn.Linear(rnn_hidden_dim + dec_hidden_dim, N_SSIG)
        self.em = EM(dict_len, rnn_hidden_dim, kernel)
        # the RL mechanism
        self.weight_init_scheme = weight_init_scheme
        self.init_model()

    def init_model(self):
        # add name fields
        self.n_ssig = N_SSIG
        self.n_vsig = N_VSIG
        self.vsig_names = vector_signal_names
        self.ssig_names = scalar_signal_names
        # init params
        initialize_weights(self, self.weight_init_scheme)

    def get_init_states(self, scale=.1, device='cpu'):
        h_0_ = sample_random_vector(self.rnn_hidden_dim, scale)
        c_0_ = sample_random_vector(self.rnn_hidden_dim, scale)
        return (h_0_, c_0_)

    def forward(self, x_t, hc_prev, beta=1):
        if self.add_query_indicator:
            x_t = add_query_indicator(x_t)
        # unpack activity
        x_t = x_t.view(1, 1, -1)
        (h_prev, c_prev) = hc_prev
        h_prev = h_prev.view(h_prev.size(1), -1)
        c_prev = c_prev.view(c_prev.size(1), -1)
        x_t = x_t.view(x_t.size(1), -1)
        # transform the input info
        preact = self.i2h(x_t) + self.h2h(h_prev)
        # get all gate values
        gates = preact[:, : N_VSIG * self.rnn_hidden_dim].sigmoid()
        c_t_new = preact[:, N_VSIG * self.rnn_hidden_dim + N_SSIG:].tanh()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.rnn_hidden_dim]
        o_t = gates[:, self.rnn_hidden_dim:2 * self.rnn_hidden_dim]
        i_t = gates[:, -self.rnn_hidden_dim:]
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(c_prev, f_t) + torch.mul(i_t, c_t_new)
        # make 1st decision attempt
        h_t = torch.mul(o_t, c_t.tanh())
        dec_act_t = F.relu(self.ih(h_t))
        # recall / encode
        hpc_input_t = torch.cat([c_t, dec_act_t], dim=1)
        inps_t = sigmoid(self.hpc(hpc_input_t))
        # [inps_t, comp_t] = torch.squeeze(phi_t)
        m_t, ma_t = self.recall(c_t, inps_t)
        cm_t = c_t + m_t
        self.encode(cm_t)
        # make final dec
        h_t = torch.mul(o_t, cm_t.tanh())
        dec_act_t = F.relu(self.ih(h_t))
        pi_a_t = _softmax(self.actor(dec_act_t), beta)
        value_t = self.critic(dec_act_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        cm_t = cm_t.view(1, cm_t.size(0), -1)
        # scache results
        scalar_signal = [inps_t, 0, 0]
        vector_signal = [f_t, i_t, o_t]
        misc = [h_t, m_t, cm_t, dec_act_t, self.em.get_vals(), ma_t]
        cache = [vector_signal, scalar_signal, misc]
        return pi_a_t, value_t, (h_t, cm_t), cache

    def recall(self, c_t, inps_t, comp_t=None):
        """run the "pattern completion" procedure

        Parameters
        ----------
        c_t : torch.tensor, vector
            cell state
        leak_t : torch.tensor, scalar
            LCA param, leak
        comp_t : torch.tensor, scalar
            LCA param, lateral inhibition
        inps_t : torch.tensor, scalar
            LCA param, input strength / feedforward weights

        Returns
        -------
        tensor, tensor
            updated cell state, recalled item

        """
        if comp_t is None:
            comp_t = self.cmpt

        if self.em.retrieval_off:
            m_t, ma_t = torch.zeros_like(c_t), None
        else:
            # retrieve memory
            m_t, ma_t = self.em.get_memory(c_t, leak=0, comp=comp_t, w_input=inps_t)
        return m_t, ma_t

    def encode(self, cm_t):
        if not self.em.encoding_off:
            self.em.save_memory(cm_t)

    def init_rvpe(self):
        self.rewards = []
        self.values = []
        self.probs = []
        self.ents = []

    def append_rvpe(self, r_t, v_t, p_a_t, ent_t):
        self.rewards.append(r_t)
        self.values.append(v_t)
        self.probs.append(p_a_t)
        self.ents.append(ent_t)

    def compute_returns(self, gamma=0, normalize=False):
        """compute return in the standard policy gradient setting.

        Parameters
        ----------
        rewards : list, 1d array
            immediate reward at time t, for all t
        gamma : float, [0,1]
            temporal discount factor, set to zero to get the actual r_t
        normalize : bool
            whether to normalize the return
            - default to false, because we care about absolute scales

        Returns
        -------
        1d torch.tensor
            the sequence of cumulative return
        """
        # compute cumulative discounted reward since t, for all t
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # normalize w.r.t to the statistics of this trajectory
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns


    def compute_a2c_loss(self, gamma=0, normalize=True, use_V=True):
        """compute the objective node for policy/value networks

        Parameters
        ----------
        probs : list
            action prob at time t
        values : list
            state value at time t
        returns : list
            return at time t

        Returns
        -------
        torch.tensor, torch.tensor
            Description of returned object.

        min sum (-pi (R_t-v_t))

        """
        returns = self.compute_returns(gamma=gamma, normalize=normalize)
        policy_grads, value_losses = [], []
        for prob_t, v_t, R_t in zip(self.probs, self.values, returns):
            if use_V:
                A_t = R_t - v_t.item()
                value_losses.append(smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t)))
            else:
                A_t = R_t
                value_losses.append(torch.FloatTensor(0).data)
            # accumulate policy gradient
            policy_grads.append(-prob_t * A_t)
        policy_gradient = torch.stack(policy_grads).sum()
        value_loss = torch.stack(value_losses).sum()
        pi_ent = torch.stack(self.ents).sum()
        # if return_all:
        return policy_gradient, value_loss, pi_ent
        # return loss_actor + loss_critic - pi_ent * eta


    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)

        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)

        """
        m = Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

    def init_em_config(self):
        self.flush_episodic_memory()
        self.encoding_off()
        self.retrieval_off()

    def flush_episodic_memory(self):
        self.em.flush()

    def encoding_off(self):
        self.em.encoding_off = True

    def retrieval_off(self):
        self.em.retrieval_off = True

    def encoding_on(self):
        self.em.encoding_off = False

    def retrieval_on(self):
        self.em.retrieval_off = False


def sample_random_vector(n_dim, scale=.1):
    return torch.randn(1, 1, n_dim) * scale


def _softmax(z, beta):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    # softmax the input to a valid PMF
    pi_a = F.softmax(torch.squeeze(z / beta), dim=0)
    # make sure the output is valid
    if torch.any(torch.isnan(pi_a)):
        raise ValueError(f'Softmax produced nan: {z} -> {pi_a}')
    return pi_a
