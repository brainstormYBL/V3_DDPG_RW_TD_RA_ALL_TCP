import numpy as np
import torch
import random

from AGENT.NET import Actor, Critic


class AGENT:
    def __init__(self, par, env, viz, index_loss_ac, index_loss_cr):
        self.par = par
        self.env = env
        self.viz = viz
        self.index_loss_ac = index_loss_ac
        self.index_loss_cr = index_loss_cr
        self.eval_actor, self.target_actor = (Actor(self.par, self.env),
                                              Actor(self.par, self.env))
        self.eval_critic, self.target_critic = (Critic(self.env, self.par),
                                                Critic(self.env, self.par))

        # self.learn_step_counter = 0
        self.memory_counter = 0
        self.buffer = []

        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.eval_actor.parameters(), lr=self.par.lr_ac)
        self.critic_optim = torch.optim.Adam(self.eval_critic.parameters(), lr=self.par.lr_cr)
        self.epi_de_flag = self.par.epi_de_flag
        self.epsilon = self.par.epsilon

    def choose_action(self, state):
        if self.epi_de_flag:
            self.epsilon += (1 - self.par.epsilon) / (self.par.max_episode * self.par.num_slot)
        if np.random.uniform() > self.epsilon:
            action = np.zeros(self.env.dim_action)
            index_action = 0
            action[index_action:index_action + self.env.num_rw_uav] = np.random.uniform(-self.par.ac_max, +self.par.ac_max, size=self.env.num_rw_uav)
            index_action += self.env.num_rw_uav
            action[index_action:index_action + self.env.num_rw_uav] = np.random.uniform(-self.par.ac_max, +self.par.ac_max, size=self.env.num_rw_uav)
            index_action += self.env.num_rw_uav
            action[index_action:index_action + self.env.num_rw_uav] = np.random.uniform(self.par.h_min, self.par.h_max, size=self.env.num_rw_uav)
            index_action += self.env.num_rw_uav
            action[index_action:index_action + self.env.num_rw_uav] = self.generate_random_action(self.env.num_rw_uav)
            index_action += self.env.num_rw_uav
            action[index_action:index_action + self.env.num_rw_uav] = self.generate_random_action(self.env.num_rw_uav)
            index_action += self.env.num_rw_uav
            action[index_action:index_action + self.env.num_rw_uav] = self.generate_random_action(self.env.num_rw_uav)
            index_action += self.env.num_rw_uav
            action[index_action:index_action + self.env.num_sub_carrier_each_uav] = self.generate_random_action(self.env.num_sub_carrier_each_uav)
            index_action += self.env.num_sub_carrier_each_uav
            action[index_action:index_action + self.env.num_sub_carrier_each_uav] = self.generate_random_action(self.env.num_sub_carrier_each_uav)
            index_action += self.env.num_sub_carrier_each_uav
            action[index_action:index_action + self.env.num_sub_carrier_each_uav] = self.generate_random_action(self.env.num_sub_carrier_each_uav)
            index_action += self.env.num_sub_carrier_each_uav
        else:
            inputs = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action = self.eval_actor(inputs).squeeze(0)
            action = action.detach().numpy()
        return action

    def store_transition(self, *transition):
        if len(self.buffer) == self.par.memory_capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):

        if len(self.buffer) < self.par.batch_size:
            return

        samples = random.sample(self.buffer, self.par.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.par.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.mean(s0)
        b = torch.std(s0)
        c = 0

        def critic_learn():
            a1 = self.target_actor(s1).detach()
            y_true = r1 + self.par.gamma * self.target_critic(s1, a1).detach()

            y_pred = self.eval_critic(s0, a0)

            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            # if self.par.visdom_flag:
            #     self.viz.line(X=[self.index_loss_cr + 1], Y=[loss.item()], win='loss of critic',
            #                   opts={'title': 'loss of critic'},
            #                   update='append')
            #     self.index_loss_cr += 1
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.eval_critic(s0, self.eval_actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            # if self.par.visdom_flag:
            #     self.viz.line(X=[self.index_loss_ac + 1], Y=[loss.item()], win='loss of actor',
            #                   opts={'title': 'loss of actor'},
            #                   update='append')
            #     self.index_loss_ac += 1
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.target_critic, self.eval_critic, self.par.tau)
        soft_update(self.target_actor, self.eval_actor, self.par.tau)

    @staticmethod
    def generate_random_action(num):
        action_ran = np.zeros(num)
        action_ran[0] = np.random.random()
        for index_uav in range(1, num):
            temp = 1 - np.sum(action_ran)
            if index_uav == num - 1:
                action_ran[index_uav] = temp
            else:
                action_ran[index_uav] = np.random.random() * temp
        return action_ran
