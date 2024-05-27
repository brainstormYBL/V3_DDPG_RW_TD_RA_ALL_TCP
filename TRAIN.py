# import gym
import random

import numpy as np
import torch
import visdom
from scipy.io import savemat

from AGENT import AGENT
from ENV.ENV import ENV
from UTILS.parameters import define_parameters


def sampling(agent, env, batch_size, viz):
    for index_epi in range(55):
        state_now = env.reset()
        index_step = 0
        # Interation in each episode until the current episode is finished (done = True)
        while True:
            # Choose an action based on the state
            action = agent.choose_action(state_now)
            # Obtain the next state, reward and done.
            state_next, reward, rate, _, done = env.step(action, index_step, viz)
            index_step += 1
            # Store the sample
            agent.store_transition(state_now, action, reward, state_next)
            # Update the state
            state_now = state_next
            if done:
                break
    samples = random.sample(agent.buffer, batch_size)
    s0, _, _, _ = zip(*samples)
    s0 = torch.tensor(s0, dtype=torch.float)
    mean = torch.mean(s0)
    std = torch.std(s0)
    return mean, std


def train(par, agent, env, viz):
    # Start interation -- episode interation
    reward_store = []
    action_store = []
    delay_s = []
    tra_rw = np.zeros((par.num_rw_uav, par.num_slot, 3))
    tra_rw_store = np.zeros((par.max_episode, par.num_rw_uav, par.num_slot, 3))
    vel_rw = np.zeros((par.num_rw_uav, par.num_slot, 2))
    vel_rw_store = np.zeros((par.max_episode, par.num_rw_uav, par.num_slot, 2))
    rate_sum = []
    delay_epi = 0
    for index_epi in range(par.max_episode):
        reward_epi = 0  # The reward in each episode
        rate_epi = 0
        # print("The training in the " + str(index_epi) + " episode is beginning.")
        # Init the environment, return the init state
        state_now = env.reset()
        index_step = 0
        # Interation in each episode until the current episode is finished (done = True)
        while True:
            # Choose an action based on the state
            action = agent.choose_action(state_now)
            # Obtain the next state, reward and done.
            state_next, reward, rate, delay, done = env.step(action, index_step, viz)
            rate_epi += rate
            delay_epi += delay
            for index_uav in range(1, par.num_rw_uav + 1):
                tra_rw[index_uav - 1][index_step] = state_next[5 * (index_uav - 1):5 * (index_uav - 1) + 3]
                vel_rw[index_uav - 1][index_step] = state_next[5 * (index_uav - 1) + 3:5 * (index_uav - 1) + 5]
            tra_rw_store[index_epi] = tra_rw
            vel_rw_store[index_epi] = vel_rw
            index_step += 1
            reward_epi += reward
            # Store the sample
            agent.store_transition(state_now, action, reward, state_next)
            # Update the state
            state_now = state_next
            # Learn
            agent.learn()
            if done:
                # print("The reward at episode " + str(index_epi) + " is " + str(reward_epi))
                # print("The action at episode " + str(index_epi) + " is " + str(action))
                reward_store.append(reward_epi)
                delay_s.append(delay_epi)
                rate_sum.append(rate_epi)
                action_store.append(action)
                if par.visdom_flag:
                    viz.line(X=[index_epi + 1], Y=[reward_epi], win='reward', opts={'title': 'reward'},
                             update='append')
                if par.visdom_flag:
                    viz.line(X=[index_epi + 1], Y=[rate_epi], win='rate', opts={'title': 'rate'},
                             update='append')
                break
    id_max = np.argmax(np.array(reward_store))
    tra_rw_res = np.array(tra_rw_store[id_max])
    vel_res = np.array(vel_rw_store[id_max])
    delay_f = delay_s[id_max]
    mat_data = {'X': tra_rw_res}
    # 保存数组到.mat文件
    savemat('tra_rw_res_frame1.mat', mat_data)

    mat_data = {'X': vel_res}
    # 保存数组到.mat文件
    savemat('vel_res_frame1.mat', mat_data)
    a = rate_sum
    b = np.sum(delay_s)
    return reward_store, tra_rw_res, vel_res


# if __name__ == '__main__':
#     # 1. Define the parameters
#     par = define_parameters()
#     viz = None
#     index_loss_ac = 0
#     index_loss_cr = 0
#     if par.visdom_flag:
#         viz = visdom.Visdom()
#         viz.close()
#     # 2. Create the environment
#     par.mean = 1
#     par.std = 1
#     env = ENV(par)
#     agent_test = AGENT.AGENT(par, env, viz, index_loss_ac, index_loss_cr)
#     mean_res, std_res = sampling(agent_test, env, par.batch_size)
#     par.mean = mean_res
#     par.std = std_res
#     # 3. Create the agent
#     agent = AGENT.AGENT(par, env, viz, index_loss_ac, index_loss_cr)
#     # 4. Start training
#     action_, tra_, vel_ = train()
#     a = 0
