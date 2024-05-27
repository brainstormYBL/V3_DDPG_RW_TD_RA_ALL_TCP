"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Train the model by the FedAvg
"""
import argparse
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import visdom

import torch

import torch.nn as nn
def parameters1():
    para = argparse.ArgumentParser()
    # Related to the FL
    para.add_argument('--num_client', type=int, default=20, help="The maximum number of the clients")
    para.add_argument('--frac', type=float, default=0.1, help="The ratio of the selected clients")
    para.add_argument('--flag', type=bool, default=False, help="The flag to express if all the client are selected")
    # Related to the training
    para.add_argument('--epochs', type=int, default=200, help="The maximum epochs for the FL training")
    para.add_argument('--epochs_local', type=int, default=5, help="The maximum epochs for the local training")
    para.add_argument('--lr', type=float, default=1e-3, help="The learning rate")
    para.add_argument('--batch_size', type=int, default=32, help="The bath size used for the local training")
    para.add_argument('--ratio_train', type=float, default=0.8, help="The ratio of the date used to train")
    para.add_argument('--visdom', type=bool, default=True, help="Open the visdom")
    para.add_argument('--model_name', type=str, default="LR", help="The name of the model")
    para.add_argument('--loss_func', default=torch.nn.MSELoss(), help="The loss function of the local training.")
    para.add_argument('--num_rw_uav', type=int, default=3, help='The number of agents')
    para.add_argument('--num_sub_carrier_each_uav', type=int, default=10, help='The init center of FW-UAV')
    para.add_argument('--num_fe_min', type=int, default=10, help='The init center of FW-UAV')
    para.add_argument('--num_fe_max', type=int, default=2000, help='The init center of FW-UAV')
    args = para.parse_args()
    return args

class LinearRegression(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LinearRegression, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.network = nn.Sequential(nn.Linear(self.dim_input, 360),
                                     nn.Linear(360, self.dim_output))

    def forward(self, x):
        output = self.network(x)
        return output


class Client:
    def __init__(self, args):
        self.args = args

    '''
    # 输入：当前Client本地数据集的特征与标签，本地模型 训练一个Client调用一次，可以采取并行多个同时计算
    # 输出：本地模型经过多次训练后的模型参数 训练过程中的平均Loss
    # 功能：训练本地模型
    '''

    def train_local(self, feature, label, model):
        num_data = feature.shape[0]
        loss_fun = self.args.loss_func  # 本地模型训练的损失函数
        optimizer = torch.optim.Adam(model.parameters(), self.args.lr)  # 本地模型训练的优化器
        loss_re = []
        # 本地多次训练 减少通信的场景
        for index_echo in range(self.args.epochs_local):
            data_selected_idx = np.array(np.random.choice(np.linspace(0, num_data - 1, num_data), self.args.batch_size)
                                         , dtype=int)
            feature_train = feature[data_selected_idx]
            label_train = torch.reshape(label[data_selected_idx], (-1, 1))
            # for param in model.parameters():
            #     print(f"Parameter Data Type: {param.dtype}")
            # print(feature_train.dtype)
            # 类型转换 转换为与模型参数一致的类型 一般为float32
            feature_train = feature_train.to(torch.float32)
            label_train = label_train.to(torch.float32)
            # 预测
            y_pre = model(feature_train)
            loss_val = loss_fun(label_train, y_pre)
            loss_re.append(loss_val.item())
            # print(loss_val.item())
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        return model.state_dict(), sum(loss_re) / len(loss_re)


class Server(nn.Module):
    def __init__(self, model_name, dim_input, dim_output, args):
        super(Server, self).__init__()
        self.model_name = model_name  # 训练的模型的名称
        self.dim_input = dim_input  # 输入的维度 即特征维度
        self.dim_output = dim_output  # 输出的维度 即标签维度
        self.args = args  # 参数列表 即为Utils中parameter函数中的内容
        self.global_model = self.build_model()  # 构建全局模型
        self.idx_client_selected = self.select_clients_join_fl()
        self.num_client_selected = len(self.idx_client_selected)

    '''
    # 输入：无
    # 输出：全局模型
    # 功能：根据模型名称构建全局模型，未识别输出None
    '''

    def build_model(self):
        if self.model_name == "LR":
            model = LinearRegression(self.dim_input, self.dim_output)
        else:
            print("模型识别出错，未定义该模型，程序退出，请重新输入！！！\n")
            sys.exit()
        return model

    '''
    # 输入：无
    # 输出：被选择参与联邦学习的Client的ID
    # 功能：确定参与联邦学习的Client
    '''

    def select_clients_join_fl(self):
        # 所有的Client均被训中
        if self.args.flag:
            idx_selected_clients = np.array(np.linspace(0, self.args.num_client - 1, self.args.num_client), dtype=int)
        # 按比例选中Client，比例为self.args.frac
        else:
            num_selected = int(self.args.num_client * self.args.frac)
            idx_selected_clients = np.random.choice(np.array(np.linspace(0, self.args.num_client - 1,
                                                                         self.args.num_client), dtype=int),
                                                    num_selected)
        return idx_selected_clients

    '''
    # 输入：无
    # 输出：全局模型的参数，字典
    # 功能：获取全局模型参数
    '''

    def get_parameters_global_model(self):
        return self.global_model.state_dict()

    '''
    # 输入：所有的本地参数
    # 输出：计算后的全局模型参数
    # 功能：计算全局模型参数
    '''

    @staticmethod
    def calculate_newest_parameters_global_model(w_local):
        # 深度复制第一个Client的网络参数
        w_avg = copy.deepcopy(w_local["mu1"])
        # keys是网络参数的键，因为是字典，例如network.0.weight表示第一层网络的权重
        for k in w_avg.keys():
            for i in range(1, len(w_local)):
                w_avg[k] += w_local["mu" + str(i + 1)][k]
            w_avg[k] = torch.div(w_avg[k], len(w_local))
        return w_avg

    @staticmethod
    def calculate_newest_parameters_global_model_fw(w_local):
        # 深度复制第一个Client的网络参数
        w_avg = copy.deepcopy(w_local["rw-uav1"])
        # keys是网络参数的键，因为是字典，例如network.0.weight表示第一层网络的权重
        for k in w_avg.keys():
            for i in range(1, len(w_local)):
                w_avg[k] += w_local["rw-uav" + str(i + 1)][k]
            w_avg[k] = torch.div(w_avg[k], len(w_local))
        return w_avg

    '''
    # 输入：最新的全局模型参数
    # 输出：无
    # 功能：更新全局模型参数
    '''

    def load_parameters_to_global_model(self, newest_par):
        self.global_model.load_state_dict(newest_par)


def ml_hfl(num_rw_uav, num_sub_carrier_each_uav, num_data_uav_mu, viz):
    # 联邦学习过程 #
    # 1. 初始化 参数 Server Client 在线打印窗口
    par1 = parameters1()
    # viz = None
    # if par1.visdom:
    #     viz = visdom.Visdom()
    #     viz.close()
    path_data_set = (r"/Users/ybl/Desktop/3.SimulationProject/3.RW_UAV_TD_AND_FW_UAV_CoRA/V3_DDPG_RW_TD_RA_ALL/5.MLFL"
                     r"/Data/heart.csv")
    data_pd = pd.read_csv(path_data_set)
    data_np = data_pd.to_numpy()
    dim_feature = 13
    dim_label = 1
    # data_holder = ProcessData(path_data_set, par1.ratio_train, par1.num_client)
    worker_mu = Client(par1)
    server_rw_uav = dict()
    for index_uav in range(num_rw_uav):
        server_rw_uav["rw-uav" + str(index_uav + 1)] = Server(par1.model_name, dim_feature, dim_label, par1)
    server_fw_uav = Server(par1.model_name, dim_feature, dim_label, par1)
    num_data = data_np.shape[0]
    data_list = np.linspace(0, num_data - 1, num_data, dtype=int)
    # 2. 进行多次的FL
    loss_dis_train = np.zeros((num_rw_uav, par1.epochs))
    loss_rw_uav = np.zeros((num_rw_uav, par1.epochs))
    loss_fw_uav = np.zeros(par1.epochs)
    fea_each_mu = dict()
    lab_each_mu = dict()
    for index_uav in range(num_rw_uav):
        fea_each_mu["rw-uav" + str(index_uav + 1)] = dict()
        lab_each_mu["rw-uav" + str(index_uav + 1)] = dict()
    for index_uav in range(num_rw_uav):
        fea_mu = dict()
        lab_mu = dict()
        for index_mu in range(num_sub_carrier_each_uav):
            index_data = np.random.choice(data_list, size=int(num_data_uav_mu[index_uav, index_mu]))
            fea_cu_mu = data_np[index_data, 0:dim_feature]
            lab_cu_mu = data_np[index_data, dim_feature:dim_feature + dim_label]
            fea_mu["mu" + str(index_mu + 1)] = fea_cu_mu
            lab_mu["mu" + str(index_mu + 1)] = lab_cu_mu
        fea_each_mu["rw-uav" + str(index_uav + 1)] = fea_mu
        lab_each_mu["rw-uav" + str(index_uav + 1)] = lab_mu
    for index_fl in range(par1.epochs):
        print("-------------开始第 " + str(index_fl + 1) + " 个epoch的训练-------------\n")
        # 2.1 本地模型获取最新的全局模型参数
        w_worker_mu_newest = dict()
        for index_uav in range(par1.num_rw_uav):
            w_worker = dict()
            for index_mu in range(par1.num_sub_carrier_each_uav):
                w_worker["mu" + str(index_mu + 1)] = server_rw_uav[
                    "rw-uav" + str(index_uav + 1)].get_parameters_global_model()
            w_worker_mu_newest["rw-uav" + str(index_uav + 1)] = w_worker
        # 2.2 针对每个Client并行执行1-5个Echos，并进行梯度下降，更新参数，并返回各自最新的参数
        loss_value = np.zeros((par1.num_rw_uav, par1.num_sub_carrier_each_uav))
        for index_uav in range(par1.num_rw_uav):
            for index_mu in range(par1.num_sub_carrier_each_uav):
                # 使用copy.deepcopy创建了一个同参数的模型副本 copy.deepcopy是深度拷贝 与原始对象相互独立
                w_worker_mu_newest["rw-uav" + str(index_uav + 1)]["mu" + str(index_mu + 1)], loss_value[index_uav][
                    index_mu] = worker_mu.train_local(
                    torch.tensor(fea_each_mu["rw-uav" + str(index_uav + 1)]["mu" + str(index_mu + 1)]),
                    torch.tensor(lab_each_mu["rw-uav" + str(index_uav + 1)]["mu" + str(index_mu + 1)]),
                    copy.deepcopy(server_rw_uav["rw-uav" + str(index_uav + 1)].global_model))
            loss_dis_train[index_uav, index_fl] = np.sum(loss_value[index_uav]) / len(loss_value[index_uav])
            radio = num_data_uav_mu[index_uav] / np.sum(num_data_uav_mu[index_uav])
            loss_rw_uav[index_uav, index_fl] = np.sum(radio * loss_value[index_uav])
        radio1 = np.sum(num_data_uav_mu, 1) / np.sum(num_data_uav_mu)
        for index_uav in range(par1.num_rw_uav):
            loss_fw_uav[index_fl] += radio1[index_uav] * loss_rw_uav[index_uav][index_fl]
        if par1.visdom:
            for index_uav in range(par1.num_rw_uav):
                viz.line(X=[index_fl + 1], Y=[loss_dis_train[index_uav][index_fl]],
                         win='MUs loss of the RW-UAV' + str(index_uav + 1),
                         opts={
                             'title': 'The average training Loss of the MUs served by the RW-UAV' + str(index_uav + 1)},
                         update='append')
                viz.line(X=[index_fl + 1], Y=[loss_rw_uav[index_uav][index_fl]],
                         win='loss of the RW-UAV' + str(index_uav + 1),
                         opts={
                             'title': 'The training Loss of the RW-UAV' + str(index_uav + 1)},
                         update='append')
                viz.line(X=[index_fl + 1], Y=[loss_fw_uav[index_fl]],
                         win='loss of the FW-UAV',
                         opts={
                             'title': 'The training Loss of the FW-UAV'},
                         update='append')
        # 2.3 根据本地参数进行全局模型参数的聚合 聚合方式很多 这里采用平均
        w_global_newest_rw = dict()
        for index_uav in range(par1.num_rw_uav):
            w_local = w_worker_mu_newest["rw-uav" + str(index_uav + 1)]
            w_global_newest = server_rw_uav["rw-uav" + str(index_uav + 1)].calculate_newest_parameters_global_model(
                w_local)
            w_global_newest_rw["rw-uav" + str(index_uav + 1)] = w_global_newest
            # 2.4 更新全局模型参数
            server_rw_uav["rw-uav" + str(index_uav + 1)].load_parameters_to_global_model(w_global_newest)
        # 2.4 第二次全局模型聚合
        w_global_newest_fw = server_fw_uav.calculate_newest_parameters_global_model_fw(w_global_newest_rw)
        server_fw_uav.load_parameters_to_global_model(w_global_newest_fw)
        # 3. 在当前的模型和数据集下，分别计算RW-UAVs和FW-UAV模型的损失函数
        print("-------------第 " + str(index_fl + 1) + " 个epoch训练结束 " + "-------------\n")
    # # 3. 画图 LOSS
    # plt.figure()
    # plt.plot(loss_dis)
    # plt.show()
    # # 4.保存模型参数
    # for index_uav in range(par1.num_rw_uav):
    #     torch.save(server_rw_uav.global_model["rw-uav" + str(index_uav + 1)].state_dict(), 'model.pth')


# if __name__ == "__main__":
#     # 准备每次FL的数据集
#     num_rw_uav = 3
#     num_sub_carrier_each_uav = 10
#     num_data_uav_mu = np.zeros((num_rw_uav, num_sub_carrier_each_uav))
#     par = parameters1()
#     fea_each_mu = dict()
#     lab_each_mu = dict()
#     for index_uav in range(num_rw_uav):
#         fea_mu = dict()
#         lab_mu = dict()
#         for index_mu in range(num_sub_carrier_each_uav):
#             num_data = np.random.randint(par.num_fe_min, par.num_fe_max)
#             num_data_uav_mu[index_uav, index_mu] = num_data
#     ml_hfl(num_rw_uav, num_sub_carrier_each_uav, num_data_uav_mu)
