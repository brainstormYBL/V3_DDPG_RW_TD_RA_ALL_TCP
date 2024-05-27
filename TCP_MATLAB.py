# -*- coding : utf-8-*-
import json
# 导入套接字模块
import socket
# 导入线程模块
import threading
import time

import numpy as np
import visdom

from AGENT import AGENT
from ENV.ENV import ENV
from TRAIN import train, sampling
from UTILS.parameters import define_parameters


def generate_mu(num_min, num_max, num_mu):
    num_fe = np.random.randint(num_min, num_max, num_mu)
    return num_fe


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


# 定义个函数,使其专门重复处理客户的请求数据（也就是重复接受一个用户的消息并且重复回答，直到用户选择下线）
def dispose_client_request(tcp_client_obj_):
    # 5 循环接收和发送数据
    data_str = ''
    while True:
        re = tcp_client_obj_.recv(1024).decode()
        data_str = data_str + re
        if "stop" in data_str:
            break
    data = json.loads(data_str)
    fw_uav_center = np.array(data['fw_uav_center'])
    fw_uav_radius = np.array(data['fw_uav_radius'])
    fw_uav_speed = np.array(data['fw_uav_speed'])
    pow_max_fw_uav = np.array(data['pow_max_fw_uav'])
    fre_fw_uav = np.array(data['fre_fw_uav'])
    cap_fw_uav = np.array(data['cap_fw_uav'])
    # 6.2 RW-UAVs 相关数据
    num_rw_uav = np.array(data['num_rw_uav'])
    pos_rw_uav_init = np.array(data['pos_rw_uav_init'])
    speed_rw_uav_init = np.array(data['speed_rw_uav_init'])
    ac_max = np.array(data['ac_max'])
    h_max = np.array(data['h_max'])
    h_min = np.array(data['h_min'])
    bw_ul_max = np.array(data['bw_ul_max'])
    bw_dl_max = np.array(data['bw_dl_max'])
    power_rw_uav_max = np.array(data['power_rw_uav_max'])
    num_sub_carrier_each_uav = np.array(data['num_sub_carrier_each_uav'])
    radius_cover_rw = np.array(data['radius_cover_rw'])
    bw_sub_carrier = np.array(data['bw_sub_carrier'])
    fre_rw_uav = np.array(data['fre_rw_uav'])
    cap_rw_uav = np.array(data['cap_rw_uav'])
    # 6.3 通信参数
    num_slot = np.array(data['num_slot'])
    length_slot = np.array(data['length_slot'])
    channel_g = np.array(data['channel_g'])
    channel_f = np.array(data['channel_f'])
    fre_cwave = np.array(data['fre_cwave'])
    epi_1 = np.array(data['epi_1'])
    epi_2 = np.array(data['epi_2'])
    epi_3 = np.array(data['epi_3'])
    beta = np.array(data['beta'])
    noise_den = np.array(data['noise_den'])
    # 6.4 MUs参数
    num_mu = np.array(data['num_mu'])
    fre_mu = np.array(data['fre_mu'])
    cap_mu = np.array(data['cap_mu'])
    power_mu = np.array(data['power_mu'])
    num_fe_min = np.array(data['num_fe_min'])
    num_fe_max = np.array(data['num_fe_max'])
    tra_mu = np.array(data['tra_mu'])

    # 6.4 DRL 参数
    max_episode = np.array(data['max_episode'])
    max_step = np.array(data['max_step'])
    size_net = np.array(data['size_net'])

    # 参数定义
    par = define_parameters()
    par.fw_uav_center = fw_uav_center
    par.fw_uav_radius = fw_uav_radius.item()
    par.fw_uav_speed = fw_uav_speed.item()
    par.pow_max_fw_uav = pow_max_fw_uav.item()
    par.fre_fw_uav = fre_fw_uav.item()
    par.cap_fw_uav = cap_fw_uav.item()
    par.num_rw_uav = num_rw_uav.item()
    par.pos_rw_uav_init = pos_rw_uav_init
    par.speed_rw_uav_init = speed_rw_uav_init
    par.ac_max = ac_max.item()
    par.h_max = h_max.item()
    par.h_min = h_min.item()
    par.bw_ul_max = bw_ul_max.item()
    par.bw_dl_max = bw_dl_max.item()
    par.power_rw_uav_max = power_rw_uav_max.item()
    par.num_sub_carrier_each_uav = num_sub_carrier_each_uav.item()
    par.radius_cover_rw = radius_cover_rw.item()
    par.bw_sub_carrier = bw_sub_carrier.item()
    par.fre_rw_uav = fre_rw_uav.item()
    par.cap_rw_uav = cap_rw_uav.item()
    par.num_slot = num_slot.item()
    par.length_slot = length_slot.item()
    par.channel_g = channel_g.item()
    par.channel_f = channel_f.item()
    par.fre_cwave = fre_cwave.item()
    par.epi_1 = epi_1.item()
    par.epi_2 = epi_2.item()
    par.epi_3 = epi_3.item()
    par.beta = beta.item()
    par.noise_den = noise_den.item()
    par.num_mu = num_mu.item()
    par.fre_mu = fre_mu
    par.cap_mu = cap_mu
    par.power_mu = power_mu.item()
    par.num_fe_min = num_fe_min.item()
    par.num_fe_max = num_fe_max.item()
    par.max_episode = max_episode.item()
    par.max_step = max_step.item()
    par.size_net = size_net.item()
    par.tra_mu = tra_mu
    tra_fw_uav = transform_fw_uav_trajectory(par.fw_uav_radius, par.fw_uav_center, par.fw_uav_speed, par.num_slot,
                                             par.length_slot)
    par.tra_fw_uav = tra_fw_uav
    num_fe = generate_mu(par.num_fe_min, par.num_fe_max, par.num_mu)
    par.num_fe = num_fe
    par.mean = 1
    par.std = 1
    viz = None
    if par.visdom_flag:
        viz = visdom.Visdom()
        viz.close()
    # 2. Create the environment
    env = ENV(par)
    # 3. Create the agent
    index_loss_ac = 0
    index_loss_cr = 0
    # 创建智能体
    # 采样定归一化的均值与方差
    agent_test = AGENT.AGENT(par, env, viz, index_loss_ac, index_loss_cr)
    mean, std = sampling(agent_test, env, par.batch_size, viz)
    par.mean = mean
    par.std = std

    agent = AGENT.AGENT(par, env, viz, index_loss_ac, index_loss_cr)
    reward, tra, vel = train(par, agent, env, viz)

    tra_1 = np.reshape(tra[0], (-1, 3))
    tra_2 = np.reshape(tra[1], (-1, 3))
    tra_3 = np.reshape(tra[2], (-1, 3))
    vel_1 = np.reshape(vel[0], (-1, 2))
    vel_2 = np.reshape(vel[1], (-1, 2))
    vel_3 = np.reshape(vel[2], (-1, 2))
    # 7 结果打包
    # reward = reward.tolist()
    tra_list = tra.tolist()
    vel_list = vel.tolist()
    res = json.dumps([reward, tra_list, vel_list])
    tcp_client_obj_.sendall(res.encode('utf-8'))
    tcp_client_obj_.close()
    print("关闭套接字")


    # 6 接收数据解析
    # 6.1 FW-UAV 轨迹


if __name__ == '__main__':
    client_id = 0
    # 1 创建服务端套接字对象
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置端口复用，使程序退出后端口马上释放
    tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    # 2 绑定端口
    tcp_server.bind(("127.0.0.1", 12345))
    # 3 设置监听
    tcp_server.listen(128)
    # 4 循环等待客户端连接请求（也就是最多可以同时有128个用户连接到服务器进行通信）
    while True:
        tcp_client_obj, tcp_client_address = tcp_server.accept()
        print("有客户端接入:", tcp_client_address)
        # 创建多线程对象
        thread = (threading.Thread(target=dispose_client_request, args=(tcp_client_obj,)))
        # 设置守护主线程  即如果主线程结束了 那子线程中也都销毁了  防止主线程无法退出
        thread.setDaemon(True)
        # 启动子线程对象
        thread.start()
        client_id += 1
