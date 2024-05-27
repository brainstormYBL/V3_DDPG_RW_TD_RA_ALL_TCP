import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

num_1 = 256
num_2 = 128


class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std


class Actor(nn.Module):
    def __init__(self, par, env):
        super(Actor, self).__init__()
        self.par = par
        self.env = env
        self.dim_state = self.env.dim_state
        self.dim_action_td_rw = self.env.dim_action_td_rw
        self.dim_action_bw_ul = self.env.dim_action_bw_ul
        self.dim_action_bw_dl = self.env.dim_action_bw_dl
        self.dim_action_power_fw = self.env.dim_action_power_fw
        self.mean = self.par.mean
        self.std = self.par.std
        self.norm_layer = NormalizationLayer(self.mean, self.std)
        self.fc1 = nn.Linear(self.dim_state, num_1)
        self.fc2 = nn.Linear(num_1, num_2)
        self.ac_x = nn.Linear(num_2, self.dim_action_td_rw)
        self.ac_y = nn.Linear(num_2, self.dim_action_td_rw)
        self.hei = nn.Linear(num_2, self.dim_action_td_rw)
        self.action_bw_ul_ly = nn.Linear(num_2, self.dim_action_bw_ul)
        self.action_bw_dl_ly = nn.Linear(num_2, self.dim_action_bw_dl)
        self.action_power_fw_ly = nn.Linear(num_2, self.dim_action_power_fw)
        self.power_rw1_ly = nn.Linear(num_2, self.env.num_sub_carrier_each_uav)
        self.power_rw2_ly = nn.Linear(num_2, self.env.num_sub_carrier_each_uav)
        self.power_rw3_ly = nn.Linear(num_2, self.env.num_sub_carrier_each_uav)

    def forward(self, x):
        x = self.norm_layer(x)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        action_ac_x = f.tanh(self.ac_x(x)) * self.par.ac_max  # [-2pi, 2pi]
        action_ac_y = f.tanh(self.ac_y(x)) * self.par.ac_max  # [-2pi, 2pi]
        action_hei = f.tanh(self.hei(x)) * (self.par.h_max - self.par.h_min) / 2 + (self.par.h_max + self.par.h_min) / 2
        action_bw_ul = f.softmax(f.sigmoid(self.action_bw_ul_ly(x)), dim=1)
        action_bw_dl = f.softmax(f.sigmoid(self.action_bw_dl_ly(x)), dim=1)
        action_power_fw = f.softmax(f.sigmoid(self.action_power_fw_ly(x)), dim=1)
        action_power_rw1 = f.softmax(f.sigmoid(self.power_rw1_ly(x)), dim=1)
        action_power_rw2 = f.softmax(f.sigmoid(self.power_rw2_ly(x)), dim=1)
        action_power_rw3 = f.softmax(f.sigmoid(self.power_rw3_ly(x)), dim=1)
        action = torch.cat((action_ac_x, action_ac_y, action_hei, action_bw_ul, action_bw_dl, action_power_fw,
                            action_power_rw1, action_power_rw2, action_power_rw3), -1)
        return action


class Critic(nn.Module):
    def __init__(self, env, par):
        super(Critic, self).__init__()
        self.env = env
        self.par = par
        self.dim_state = self.env.dim_state
        self.dim_action = self.env.dim_action
        self.mean = self.par.mean
        self.std = self.par.std
        self.norm_layer = NormalizationLayer(self.mean, self.std)
        self.fc1 = nn.Linear(self.dim_state + self.dim_action, num_1)
        self.fc2 = nn.Linear(num_1, num_2)
        self.q_out = nn.Linear(num_2, self.dim_action)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.norm_layer(x)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value
