import torch
import torch.nn as nn

import utils


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        # inputs actions and observations to predict Q value
        self.Q = nn.Sequential(nn.Linear(repr_dim+action_shape[0], hidden_dim), 
                               nn.ReLU(inplace=True), 
                               nn.Linear(hidden_dim, hidden_dim), 
                               nn.ReLU(inplace=True), 
                               nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        # print ("obs is ", obs.size(), "\n")
        # print ("action is ", action.size(), "\n")
        # print ("concat size is ", torch.concat((obs, action), dim=-1).size(), "\n")

        q = self.Q(torch.concat((obs, action), dim=-1)) # should I constrain this value
        return q

