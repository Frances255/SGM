import torch
import torch.nn as nn


class global_attention(nn.Module):

    def __init__(self, hidden_size, activation=None):
        super(global_attention, self).__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2*hidden_size, hidden_size)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context):
        # x: batch * hidden_size
        # context: batch * time * hidden_size

        # batch * hidden_size * 1
        gamma_h = self.linear_in(x).unsqueeze(2)
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        # batch * time * hidden_size  batch * hidden_size * 1 => batch * time * 1 => batch * time
        weights = torch.bmm(context, gamma_h).squeeze(2)
        # batch * time
        weights = self.softmax(weights)
        # batch * 1 * time  batch * time * hidden_size => batch * 1 * hidden_size => batch * hidden_size
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)
        # batch * 2 * hidden_size => batch * hidden_size
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights