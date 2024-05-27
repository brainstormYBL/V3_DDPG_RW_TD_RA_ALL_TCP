import numpy as np

from MLHFL import ml_hfl


class ENV:
    def __init__(self, par):
        self.par = par
        self.num_rw_uav = self.par.num_rw_uav
        self.radius_fw_uav = self.par.fw_uav_radius
        self.center_fw_uav = self.par.fw_uav_center
        self.num_slot = self.par.num_slot
        self.time_interval = self.par.length_slot
        self.tra_fw_uav = self.par.tra_fw_uav
        self.dim_action_td_rw = self.num_rw_uav
        self.dim_action_bw_ul = self.num_rw_uav
        self.dim_action_bw_dl = self.num_rw_uav
        self.dim_action_power_fw = self.par.num_rw_uav
        self.num_sub_carrier_each_uav = self.par.num_sub_carrier_each_uav
        self.dim_action = 6 * self.num_rw_uav + self.num_rw_uav * self.num_sub_carrier_each_uav
        self.dim_state = 8 * self.num_rw_uav + 2
        self.bw_dl_max = self.par.bw_dl_max
        self.bw_ul_max = self.par.bw_ul_max
        self.length_slot = self.par.length_slot
        self.pow_max_fw_uav = self.par.pow_max_fw_uav
        self.power_rw_uav_max = self.par.power_rw_uav_max
        self.beta = self.par.beta
        self.noise_den = self.par.noise_den
        self.num_fea = self.par.num_fe
        self.tra_mu = self.par.tra_mu
        self.radius_cover_rw = self.par.radius_cover_rw
        self.channel_g = self.par.channel_g
        self.fre_cwave = self.par.fre_cwave
        self.channel_f = self.par.channel_f
        self.power_mu = self.par.power_mu
        self.bw_sub_carrier = self.par.bw_sub_carrier
        self.size_net = self.par.size_net
        self.fre_fw_uav = self.par.fre_fw_uav
        self.cap_fw_uav = self.par.cap_fw_uav
        self.fre_rw_uav = self.par.fre_rw_uav
        self.cap_rw_uav = self.par.cap_rw_uav
        self.fre_mu = self.par.fre_mu
        self.cap_mu = self.par.cap_mu
        self.epi_1 = self.par.epi_1
        self.epi_2 = self.par.epi_2
        self.epi_3 = self.par.epi_3
        self.max_episode = self.par.max_episode
        # self.num_data_uav_mu = self.par.num_data_uav_mu

        self.pos_rw_uav_init = np.array(self.par.pos_rw_uav_init)
        self.speed_rw_uav_init = np.array(self.par.speed_rw_uav_init)
        self.pos_rw_uav_now = np.array(self.par.pos_rw_uav_init)
        self.vel_rw_uav_now = np.array(self.par.speed_rw_uav_init)

    def reset(self):
        obs_init = np.zeros(self.dim_state)
        index_obs = 0
        self.pos_rw_uav_now = np.array(self.pos_rw_uav_init)
        self.vel_rw_uav_now = np.array(self.speed_rw_uav_init)
        for index in range(self.par.num_rw_uav):
            obs_init[index_obs] = self.pos_rw_uav_init[index][0]
            index_obs += 1
            obs_init[index_obs] = self.pos_rw_uav_init[index][1]
            index_obs += 1
            obs_init[index_obs] = self.pos_rw_uav_init[index][2]
            index_obs += 1
            obs_init[index_obs] = self.speed_rw_uav_init[index][0]
            index_obs += 1
            obs_init[index_obs] = self.speed_rw_uav_init[index][1]
            index_obs += 1
        obs_init[index_obs:index_obs + 2] = self.tra_fw_uav[0][0:2]
        index_obs += 2
        # 存储覆盖范围内MUs数量
        num_mu_each_rw_uav = np.zeros(self.num_rw_uav)
        # 存储覆盖范围内MUs的样本总量
        num_fe = np.zeros(self.num_rw_uav)
        cap_mu = np.zeros(self.num_rw_uav)
        for index_uav in range(self.num_rw_uav):
            num_fe[index_uav], num_mu_each_rw_uav[index_uav], cap_mu[index_uav] = self.calculate_num_mu_each_rw_uav(
                self.pos_rw_uav_now[index_uav][0:3], self.tra_mu[0][:][:])
        obs_init[index_obs:index_obs + self.num_rw_uav] = num_mu_each_rw_uav
        index_obs += self.num_rw_uav
        obs_init[index_obs:index_obs + self.num_rw_uav] = num_fe
        index_obs += self.num_rw_uav
        obs_init[index_obs:index_obs + self.num_rw_uav] = cap_mu
        return obs_init

    def step(self, action, index_step, viz):
        done = False
        index_obs = 0
        obs_next = np.zeros(self.dim_state)
        ac_x = action[0:self.par.num_rw_uav]
        ac_y = action[self.num_rw_uav:2 * self.num_rw_uav]
        hei = action[2 * self.num_rw_uav:3 * self.num_rw_uav]
        bw_ul = action[3 * self.num_rw_uav:4 * self.num_rw_uav] * self.bw_ul_max
        bw_dl = action[4 * self.num_rw_uav:5 * self.num_rw_uav] * self.bw_dl_max
        power_rw_fw_dl = action[5 * self.num_rw_uav:6 * self.num_rw_uav] * self.pow_max_fw_uav
        power_rw_mu_dl = np.zeros((self.num_rw_uav, self.num_sub_carrier_each_uav))
        index_action = 6 * self.num_rw_uav
        for index_uav in range(self.num_rw_uav):
            power_rw_mu_dl[index_uav, :] = action[index_action + index_uav * self.num_sub_carrier_each_uav:index_action + (index_uav + 1) * self.num_sub_carrier_each_uav] # * self.power_rw_uav_max
        # 根据动作更新RW-UAVs速度 位置 得到带宽 功率分配
        pos_rw_uav_new = np.zeros((self.par.num_rw_uav, 3))
        vel_rw_uav_new = self.vel_rw_uav_now + np.reshape(np.array([ac_x, ac_y]), (-1, 2)) * self.length_slot
        pos_rw_uav_new[:, 0:2] = self.pos_rw_uav_now[:, 0:2] + vel_rw_uav_new * self.length_slot + 1/2 * np.reshape(np.array([ac_x, ac_y]), (-1, 2)) * self.length_slot ** 2
        pos_rw_uav_new[:, 2] = hei[:]
        self.vel_rw_uav_now[:, :] = vel_rw_uav_new[:, :]
        self.pos_rw_uav_now[:, 0:2] = pos_rw_uav_new[:, 0:2]
        self.pos_rw_uav_now[:, 2] = pos_rw_uav_new[:, 2]
        for index in range(self.par.num_rw_uav):
            obs_next[index_obs] = self.pos_rw_uav_now[index][0]
            index_obs += 1
            obs_next[index_obs] = self.pos_rw_uav_now[index][1]
            index_obs += 1
            obs_next[index_obs] = self.pos_rw_uav_now[index][2]
            index_obs += 1
            obs_next[index_obs] = self.vel_rw_uav_now[index][0]
            index_obs += 1
            obs_next[index_obs] = self.vel_rw_uav_now[index][1]
            index_obs += 1
        obs_next[index_obs:index_obs + 2] = self.tra_fw_uav[index_step][0:2]
        index_obs += 2
        # 存储覆盖范围内MUs数量
        num_mu_each_rw_uav = np.zeros(self.num_rw_uav)
        # 存储覆盖范围内MUs的样本总量
        num_fe = np.zeros(self.num_rw_uav)
        cap_mu = np.zeros(self.num_rw_uav)
        for index_uav in range(self.num_rw_uav):
            num_fe[index_uav], num_mu_each_rw_uav[index_uav], cap_mu[index_uav] = self.calculate_num_mu_each_rw_uav(
                self.pos_rw_uav_now[index_uav][0:3], self.tra_mu[index_step][:][:])
        obs_next[index_obs:index_obs + self.num_rw_uav] = num_mu_each_rw_uav
        index_obs += self.num_rw_uav
        obs_next[index_obs:index_obs + self.num_rw_uav] = num_fe
        index_obs += self.num_rw_uav
        obs_next[index_obs:index_obs + self.num_rw_uav] = cap_mu
        rate_ul_rw_fw, rate_dl_rw_fw = self.calculate_rate_fw_rw_uav(self.tra_fw_uav[index_step], self.pos_rw_uav_now, bw_ul, bw_dl, power_rw_fw_dl)
        rate_dl_mu_rw = np.zeros((self.num_rw_uav, self.num_sub_carrier_each_uav))
        rate_ul_mu_rw = np.zeros((self.num_rw_uav, self.num_sub_carrier_each_uav))
        mu_id = np.zeros((self.num_rw_uav, self.num_sub_carrier_each_uav), dtype=int)
        mu_id[:, :] = np.linspace(0, 9, 10, dtype=int)
        delay_total = np.zeros(self.num_rw_uav)
        computation = [1, 1, 1]
        for index_uav in range(self.num_rw_uav):
            r_ul, r_dl = self.calculate_rate_mu_rw_uav_current_pos(self.pos_rw_uav_now[index_uav],
                                                                   self.tra_mu[index_step][mu_id[index_uav, :]],
                                                                   power_rw_mu_dl[index_uav, :],
                                                                   )
            rate_ul_mu_rw[index_uav, :] = r_ul
            rate_dl_mu_rw[index_uav, :] = r_dl
            delay_total[index_uav] = self.calculate_delay_each_rw_uav(bw_dl * rate_dl_rw_fw[index_uav],
                                                                      bw_ul * rate_ul_rw_fw[index_uav],
                                                                      self.bw_sub_carrier * rate_ul_mu_rw[index_uav, :],
                                                                      self.bw_sub_carrier * rate_dl_mu_rw[index_uav, :],
                                                                      self.num_fea[mu_id[index_uav]],
                                                                      self.cap_mu[mu_id[index_uav]],
                                                                      self.fre_mu[mu_id[index_uav]],
                                                                      computation[index_uav])
        reward = (self.epi_1 * (np.sum(rate_ul_rw_fw + rate_dl_rw_fw)) - self.epi_2 * np.sum(delay_total)) / 200
        # ml_hfl(self.num_rw_uav, self.num_sub_carrier_each_uav, self.num_data_uav_mu, viz)
        # reward = (self.epi_1 * (np.sum(rate_ul_rw_fw + rate_dl_rw_fw)) - self.epi_2 * np.sum(delay_total)) / 200
        if index_step >= self.par.num_slot - 1:
            done = True
        rate_sum = np.sum(self.bw_sub_carrier * rate_ul_mu_rw + self.bw_sub_carrier * rate_dl_mu_rw)
        return obs_next, reward, np.sum(rate_sum), np.sum(delay_total), done

    def generate_rw_uav_trajectory(self):
        # The trajectory of RW-UAVs, consisting: [UAV ID, time slot index, x, y, z]
        rw_uav_trajectory = np.zeros((self.par.num_slot, self.par.num_rw_uav, 3))
        for index_slot in range(self.par.num_slot):
            for index_rw_uav in range(self.par.num_rw_uav):
                rw_uav_trajectory[index_slot, index_rw_uav, 0] = (np.random.uniform(low=index_slot * 100 - 50,
                                                                                    high=index_slot * 100 + 50)
                                                                  )
                rw_uav_trajectory[index_slot, index_rw_uav, 1] = (np.random.uniform(low=index_slot * 100 - 50,
                                                                                    high=index_slot * 100 + 50)
                                                                  ) + index_rw_uav * 800
                rw_uav_trajectory[index_slot, index_rw_uav, 2] = (index_rw_uav + 1) * 50
        return rw_uav_trajectory

    def calculate_rate_fw_rw_uav(self, tra_rw_uav, pos_rw_uav, bw_ul_al, bw_dl_al, power_al):
        dis = np.sqrt(np.sum((tra_rw_uav[0:2] - pos_rw_uav[:, 0:2]) ** 2, 1) + (tra_rw_uav[2] - pos_rw_uav[:, 2]) ** 2)
        gain = self.beta / (dis ** 2)
        # for index_rw_uav in range(self.num_rw_uav):
        #     if bw_ul_al[index_rw_uav] == 0:
        #         bw_ul_al[index_rw_uav] = self.bw_ul_max
        #     if bw_dl_al[index_rw_uav] == 0:
        #         bw_dl_al[index_rw_uav] = self.bw_dl_max
        sinr_ul = self.power_rw_uav_max * gain / (bw_ul_al * self.noise_den)
        sinr_dl = power_al * gain / (bw_dl_al * self.noise_den)
        rate_dl = np.log2(1 + sinr_dl)
        rate_ul = np.log2(1 + sinr_ul)
        return rate_ul, rate_dl

    @staticmethod
    def calculate_start_point(radius, center):
        temp = [center[0] - radius, center[1]]
        return np.array([temp[0][0], temp[1]])

    '''
    功能:计算当前RW-UAV当前时隙覆盖的MUs数量 以及覆盖MUs的样本数量之和
    输入：当前RW-UAV的位置pos_rw 维度：1*3 所有MUs的位置pos_mus 维度：N*2
    输出：样本数量之和num_fe 1*1 用户数量num_mu 1*1
    '''
    def calculate_num_mu_each_rw_uav(self, pos_rw, pos_mus):
        dis = np.sqrt(np.sum((pos_rw[0:2] - pos_mus[:, :]) ** 2, 1))
        num_mu = len(dis[dis < self.radius_cover_rw])
        num_fe = np.sum(self.num_fea[dis < self.radius_cover_rw])
        cap_mu = np.sum(self.cap_mu[dis < self.radius_cover_rw])
        return num_fe, num_mu, cap_mu

    '''
        功能:计算当前RW-UAV当前时与被调度MUs之间的通信速率
        输入：当前RW-UAV的位置pos_rw 维度：1*3 被调度MUs位置pos_mu K*2 当前RW-UAV功率分配怒情况power K*1
        输出：当前RW-UAV与被调度MUs之间的上行速率 rate_ul K*1 下行速率rate_dl K*1
        '''

    def calculate_rate_mu_rw_uav_current_pos(self, pos_rw, pos_mu, power):
        dis = np.sqrt(np.sum((pos_rw[0:2] - pos_mu[:, 0:2]) ** 2, 1) + pos_rw[2] ** 2)
        theta = np.arcsin(pos_rw[2] / dis)
        eta_los = 1 / (1 + self.channel_f * np.exp(-self.channel_g * (theta - self.channel_f)))
        eta_n_los = 1 - eta_los
        pl_los = 28 + 20 * np.log10(self.fre_cwave) + 22 * np.log10(dis)
        pl_n_los = -17.5 + 20 * np.log10(4 * np.pi * self.fre_cwave / 3) + (46 - 7 * np.log10(pos_rw[2])) * np.log10(
            dis)
        gain = eta_los * pl_los + eta_n_los * pl_n_los
        sinr_ul = self.power_mu * gain / (self.bw_sub_carrier * self.par.noise_den)
        sinr_dl = power * gain / (self.bw_sub_carrier * self.par.noise_den)
        rate_dl = self.bw_sub_carrier * np.log2(1 + sinr_dl)
        rate_ul = self.bw_sub_carrier * np.log2(1 + sinr_ul)
        # for index in range(len(power)):
        #     if rate_dl[index] <= 1:
        #         rate_dl[index] = 0.01
        #     if rate_ul[index] <= 1:
        #         rate_ul[index] = 0.01
        return rate_ul, rate_dl

    '''
        功能:计算当前RW-UAV FL的时间
        输入：两层网络之间的上下行速率 rate_fw_rw_dl 1*1 rate_fw_rw_ul 1*1 rate_rw_mu_ul 1*K rate_rw_mu_dl 1*K
        被调度的MUs的样本总数量num_fea 1*4 被调度用户的本地处理能力 cap_mu fre_mu 1*K 当前RW-UAV分得计算能力用于第二次聚合cpu_radio 1*1 
        输出：总的时间 1*1
        '''

    def calculate_delay_each_rw_uav(self, rate_fw_rw_dl, rate_fw_rw_ul, rate_rw_mu_ul, rate_rw_mu_dl, num_fea, cap_mu,
                                    fre_mu, cpu_radio):
        # 本地计算的时间
        if cpu_radio == 0:
            cpu_radio = 0.0001
        delay_local = (num_fea * cap_mu) / fre_mu
        t1 = max(delay_local)
        delay_mu_rw = np.zeros(self.num_sub_carrier_each_uav)
        delay_rw_mu = np.zeros(self.num_sub_carrier_each_uav)
        for index_mu in range(self.num_sub_carrier_each_uav):
            delay_mu_rw[index_mu] = self.calculate_num_slot_finish(rate_rw_mu_ul[index_mu])
            delay_rw_mu[index_mu] = self.calculate_num_slot_finish(rate_rw_mu_dl[index_mu])
        delay_rw_fw = np.zeros(self.num_rw_uav)
        delay_fw_rw = np.zeros(self.num_rw_uav)
        for index_uav in range(self.num_rw_uav):
            delay_rw_fw[index_uav] = self.calculate_num_slot_finish(rate_fw_rw_ul[index_uav])
            delay_fw_rw[index_uav] = self.calculate_num_slot_finish(rate_fw_rw_dl[index_uav])
        t2 = self.time_interval * max(delay_mu_rw)
        t3 = (np.sum(num_fea) * self.cap_rw_uav) / self.fre_rw_uav
        t4 = self.time_interval * max(delay_rw_fw)
        t5 = ((cpu_radio * self.cap_fw_uav) * np.sum(num_fea)) / self.fre_fw_uav
        t6 = self.time_interval * max(delay_fw_rw)
        t7 = self.time_interval * max(delay_rw_mu)
        delay_total = t1 + t2 + t3 + t4 + t5 + t6 + t7
        return delay_total

    '''
    功能:计算传完模型参数需要的时隙数
    输入：通信速率rate 1*1
    输出：时隙数 1*1
    '''

    def calculate_num_slot_finish(self, rate):
        num_slot = 1
        size_trans = 0
        while True:
            size_trans += rate
            if size_trans >= self.size_net:
                break
            num_slot += 1
        return num_slot
