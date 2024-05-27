"""
Author   : Bao-lin Yin
Data     : 2.27 2024
Version  : V1.0
Function : Defining the parameters.
"""
import argparse

import numpy as np


def transform_fw_uav_trajectory(radius, center, speed, num_slot, time_interval):
    tra = np.zeros((num_slot, 3))
    tra[0, 0] = center[0] - radius
    tra[0, 1] = center[1]
    tra[0, 2] = 1000
    # The length of the arc in each time slot, Units:rad
    theta_each_slot = (speed * time_interval) / radius
    for index_slot in range(1, num_slot):
        tra[index_slot, 0] = center[0] - radius * np.cos(index_slot * theta_each_slot)
        tra[index_slot, 1] = center[1] - radius * np.sin(index_slot * theta_each_slot)
        tra[index_slot, 2] = 1000
    return tra


def generate_mu(num_slot, num_mu, fw_uav_center, fw_uav_radius):
    tra_mu = np.zeros((num_slot, num_mu, 2))
    x_tra = np.random.uniform(fw_uav_center[0] - fw_uav_radius, fw_uav_center[0] + fw_uav_radius)
    y_tra = np.random.uniform(fw_uav_center[1] - fw_uav_radius, fw_uav_center[1] + fw_uav_radius, num_mu)
    temp = np.zeros((num_mu, 2))
    temp[:, 0] = x_tra
    temp[:, 1] = y_tra
    tra_mu[:, :, :] = temp
    num_fe = np.random.randint(30, 200, num_mu)
    return num_fe, tra_mu


def define_parameters():
    para = argparse.ArgumentParser("The parameters for solving the trajectory design of FW-UAV")
    # The parameters for the DDPG
    para.add_argument('--lr_ac', type=float, default=5e-4, help='The learning rate of actor')
    para.add_argument('--lr_cr', type=float, default=5e-4, help='The learning rate of critic')
    para.add_argument('--gamma', type=float, default=0.99, help='The discount factor')
    para.add_argument('--epsilon', type=float, default=0.98, help='The greedy factor')
    para.add_argument('--memory_capacity', type=int, default=1000, help='The size of the memory')
    para.add_argument('--batch_size', type=int, default=256, help='The batch size')
    para.add_argument('--tau', type=float, default=0.005, help='The tau')

    # The parameters for the plot
    para.add_argument('--visdom_flag', type=bool, default=True, help='visdom is enabled')

    # The parameters for the test
    para.add_argument('--epi_de_flag', type=bool, default=False, help='The init center of FW-UAV')
    args = para.parse_args()
    # tra_fw_uav = transform_fw_uav_trajectory(args.fw_uav_radius, args.fw_uav_center, args.fw_uav_speed, args.num_slot,
    #                                          args.length_slot)
    # args.tra_fw_uav = tra_fw_uav
    # num_fe, tra_mu = generate_mu(args.num_slot, args.num_mu, args.fw_uav_center, args.fw_uav_radius)
    # args.num_fe = num_fe
    # args.tra_mu = tra_mu
    # num_data_uav_mu = np.zeros((args.num_rw_uav, args.num_sub_carrier_each_uav))
    # for index_uav in range(args.num_rw_uav):
    #     for index_mu in range(args.num_sub_carrier_each_uav):
    #         num_data = np.random.randint(args.num_fe_min, args.num_fe_max)
    #         num_data_uav_mu[index_uav, index_mu] = num_data
    # args.num_data_uav_mu = num_data_uav_mu
    return args
